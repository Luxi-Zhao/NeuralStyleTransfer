import React, { useReducer, useState, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Button } from '@material-ui/core';
import CircularProgress from '@material-ui/core/CircularProgress';
import './App.css';

const stateMachine = {
  initial: 'awaitingUpload',
  states: {
    awaitingUpload: { on: { startTransfer: 'transferring' } },
    transferring: { on: { transferDone: 'complete' }, showLoadingIcon: true },
    complete: { on: { reset: 'awaitingUpload' }, showResult: true },
  }
}

const reducer = (currentState, event) => stateMachine.states[currentState].on[event] || stateMachine.initial;

const formatResult = ({ className, probability }) => (
  <li key={className}>
    {`${className}: %${(probability * 100).toFixed(2)}`}
  </li>
)

const contentLayers = [
  'conv_dw_13',
  'conv_pw_13',
];
const styleLayers = [
  'conv_dw_5',
  'conv_pw_5',
  'conv_dw_6',
  'conv_pw_6',
  'conv_dw_7',
  'conv_pw_7',
  'conv_dw_8',
  'conv_pw_8',
  'conv_dw_9',
  'conv_pw_9',
];

function App() {
  const [state, dispatch] = useReducer(reducer, stateMachine.initial);
  const [contentImageUrl, setContentImageUrl] = useState(null);
  const [styleImageUrl, setStyleImageUrl] = useState(null);
  const [results, setResults] = useState([]);
  const contentInputRef = useRef();
  const styleInputRef = useRef();
  const contentImageRef = useRef();
  const styleImageRef = useRef();
  const canvasRef = useRef();

  const handleUpload = (e, imgType) => {
    const { files } = e.target;
    if (files.length > 0) {
      const url = URL.createObjectURL(files[0]);
      if (imgType === 'content') {
        setContentImageUrl(url);
      } else if (imgType === 'style') {
        setStyleImageUrl(url);
      } else {
        console.err('invalid upload img type');
      }
    }
  }

  const handleStyleTransfer = async () => {
    console.log('handling style transfer')
    dispatch('startTransfer');
    await doTransfer();
    dispatch('transferDone');
    console.log('done')
  }

  const doTransfer = async () => {
    const model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');

    const styleImgTensor = htmlImgToTensor(styleImageRef.current);
    const contentImgTensor = htmlImgToTensor(contentImageRef.current);

    const styleExtractor = getFeatureMapExtractor(styleLayers, model);
    const contentExtractor = getFeatureMapExtractor(contentLayers, model);

    const styleTargets = getStyleOutputs(styleExtractor, styleImgTensor);
    const contentTargets = getContentOutputs(contentExtractor, contentImgTensor);

    // Performance evaluation:
    // Using await, takes 6402milsec to get style and content targets
    // Using sync, takes 6820milsec and 5422 sec to get style and content targets
    // Not much difference, use sync for easiness

    console.log('got style and content targets');

    const optimizer = tf.train.adam(0.02, 0.99, undefined, 0.1);
    const resultImage = tf.variable(contentImgTensor);

    const epochs = 5;
    const stepsPerEpoch = 10;
    let epoch, step;
    for (epoch = 0; epoch < epochs; epoch++) {
      for (step = 0; step < stepsPerEpoch; step++) {
        trainStep(
          resultImage, optimizer, styleTargets, contentTargets, styleExtractor, contentExtractor);
        console.log('.');
      }
    }
    tf.browser.toPixels(resultImage.squeeze(), canvasRef.current);
  }

  const trainStep = (image, optimizer, styleTargets, contentTargets, styleExtractor, contentExtractor) => {
    const lossFunction = () => {
      const styleOutputs = getStyleOutputs(styleExtractor, image);
      const contentOutputs = getContentOutputs(contentExtractor, image);

      const styleWeight = 1;
      const contentWeight = 1;
      let styleLoss = tf.addN(styleOutputs.map((styleOutput, index) => tf.mean(
        tf.square(tf.sub(styleOutput, styleTargets[index]))
      )));
      styleLoss = tf.mul(tf.div(styleLoss, styleLayers.length), styleWeight);

      let contentLoss = tf.addN(contentOutputs.map((contentOutput, index) => tf.mean(
        tf.square(tf.sub(contentOutput, contentTargets[index]))
      )));
      contentLoss = tf.mul(tf.div(contentLoss, contentLayers.length), contentWeight);

      const loss = tf.add(styleLoss, contentLoss);
      return loss;
    };

    const grads = tf.variableGrads(lossFunction);
    optimizer.applyGradients(grads.grads);
    image.assign(clip_0_1(image));
  }

  /**
   * @param {*} image a tensor
   */
  const clip_0_1 = image => {
    return tf.clipByValue(image, 0, 1);
  }

  // Convert image from any size to a tensor of size
  // [1, 224, 224, 3]
  const htmlImgToTensor = htmlImg => {
    let tensor = tf.browser.fromPixels(htmlImg);
    tensor = tf.div(tensor, 255.0);
    tensor = tf.image.resizeBilinear(tensor, [224, 224]);
    tensor = tf.expandDims(tensor, 0);
    return tensor;
  }

  const getFeatureMapExtractor = (layerNames, model) => {
    const outputs = layerNames.map(layerName => model.getLayer(layerName).output);
    const modifiedModel = tf.model({ inputs: model.inputs, outputs: outputs });
    return modifiedModel;
  }

  /**
   * Returns an **array** of gram matrices of shape (1, depth, depth)
   * @param {*} model 
   * @param {*} styleLayers 
   * @param {*} styleImgTensor 
   */
  const getStyleOutputs = (styleExtractor, styleImgTensor) => {
    styleImgTensor = tf.mul(styleImgTensor, 255);
    const styleFeatMaps = styleExtractor.predict(styleImgTensor);
    // Returns an **array** of tensors (FMs)
    // Each tensor is an FM of shape (1, featMapheight, FMwidth, FM set depth)
    // FMs are the resulting feature maps from passing a style image into the mobilenet model

    const gramMatrices = styleFeatMaps.map(ftmap => getGramMatrix(ftmap));
    return gramMatrices;
  }

  /**
   * Returns an **array** of feature maps of shape (1, height, width, depth)
   * @param {*} model 
   * @param {*} contentLayers 
   * @param {*} contentImgTensor 
   */
  const getContentOutputs = (contentExtractor, contentImgTensor) => {
    contentImgTensor = tf.mul(contentImgTensor, 255);
    const contentFeatMaps = contentExtractor.predict(contentImgTensor);
    return contentFeatMaps;
  }

  /**
   * Shape of inputTensor is (1, 224, 224, depth)
   * @param {*} inputTensor shape (1, FMHeight, FMWidth, numFMsInLayer/Depth)
   * Returns a gram matrix of shape (1, depth, depth)
   */
  const getGramMatrix = inputTensor => {
    const numLayers = inputTensor.shape[0];
    const height = inputTensor.shape[1];
    const width = inputTensor.shape[2];
    const depth = inputTensor.shape[3];

    inputTensor = tf.transpose(inputTensor, [0, 3, 1, 2]).reshape([numLayers, depth, height * width]);
    let gramMatrix = tf.matMul(inputTensor, tf.transpose(inputTensor, [0, 2, 1]));
    gramMatrix = tf.div(gramMatrix, height * width);
    return gramMatrix;
  }

  const handleReset = () => {
    dispatch('reset');
    setContentImageUrl(null);
    setStyleImageUrl(null);
  }

  const { showLoadingIcon, showResult } = stateMachine.states[state];


  return (
    <div className="App">
      <canvas
        id="canvas"
        ref={canvasRef}
        width={500}
        height={500}
        style={{
          border: '2px solid #000',
          marginTop: 10,
        }}
      />
      {showResult && <div>
        <ul>
          {results.map(formatResult)}
        </ul>
        <Button
          color="primary"
          aria-label="outlined primary"
          onClick={handleReset}
        >
          Reset
        </Button>
      </div>}
      {showLoadingIcon && <div>
        <CircularProgress />
      </div>}
      {!showResult && <div>
        {contentImageUrl && <img alt="content-preview" src={contentImageUrl} ref={contentImageRef} />}
        {styleImageUrl && <img alt="style-preview" src={styleImageUrl} ref={styleImageRef} />}

        <input
          id="content-file-input"
          className="file-input"
          type="file"
          accept="image/*"
          ref={contentInputRef}
          onChange={e => handleUpload(e, 'content')}
        />
        <input
          id="style-file-input"
          className="file-input"
          type="file"
          accept="image/*"
          ref={styleInputRef}
          onChange={e => handleUpload(e, 'style')}
        />
        <label htmlFor="content-file-input">
          <Button
            color="primary"
            aria-label="outlined primary"
            onClick={() => contentInputRef.current.click()}
          >
            Upload Content Image
          </Button>
        </label>
        <label htmlFor="style-file-input">
          <Button
            color="primary"
            aria-label="outlined primary"
            onClick={() => styleInputRef.current.click()}
          >
            Upload Style Image
          </Button>
        </label>
        {contentImageUrl && styleImageUrl &&
          <Button
            color="primary"
            aria-label="outlined primary"
            onClick={handleStyleTransfer}
          >
            Combine
        </Button>}
      </div>}
    </div>
  );
}

export default App;

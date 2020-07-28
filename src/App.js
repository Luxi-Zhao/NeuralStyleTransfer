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

const contentLayers = [
  'block5_conv2',
];
const styleLayers = [
  'block1_conv1',
  'block2_conv1',
  'block3_conv1',
  'block4_conv1',
  'block5_conv1'
];

function App() {
  const [state, dispatch] = useReducer(reducer, stateMachine.initial);
  const [contentImageUrl, setContentImageUrl] = useState(null);
  const [styleImageUrl, setStyleImageUrl] = useState(null);
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
    const model = await tf.loadLayersModel('http://localhost:8080/model.json');

    const styleImgTensor = htmlImgToTensor(styleImageRef.current);
    const contentImgTensor = htmlImgToTensor(contentImageRef.current);

    const styleExtractor = getFeatureMapExtractor(styleLayers, model);
    const contentExtractor = getFeatureMapExtractor(contentLayers, model);

    const styleTargets = getStyleOutputs(styleExtractor, styleImgTensor);
    const contentTargets = getContentOutputs(contentExtractor, contentImgTensor);

    console.log('got style and content targets');

    const optimizer = tf.train.adam(0.02, 0.99, undefined, 0.1);
    const resultImage = tf.variable(contentImgTensor);

    const epochs = 1;
    const stepsPerEpoch = 100;
    let epoch, step;
    for (epoch = 0; epoch < epochs; epoch++) {
      for (step = 0; step < stepsPerEpoch; step++) {
        trainStep(
          resultImage, optimizer, styleTargets, contentTargets, styleExtractor, contentExtractor);
        console.log('.');
      }
      console.log(`Epoch #${epoch} done.`)
    }
    tf.browser.toPixels(resultImage.squeeze(), canvasRef.current);
  }

  const trainStep = (image, optimizer, styleTargets, contentTargets, styleExtractor, contentExtractor) => {
    const lossFunction = () => {
      const styleOutputs = getStyleOutputs(styleExtractor, image);
      const contentOutputs = getContentOutputs(contentExtractor, image);

      const styleWeight = 100;
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

    optimizer.minimize(lossFunction, false, [image]);
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
    tensor = tf.image.resizeBilinear(tensor, [112, 112]);
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

    // When there is only one contentLayer, the predictor returns a single tensor
    if (Array.isArray(contentFeatMaps)) return contentFeatMaps;
    else return [contentFeatMaps];
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

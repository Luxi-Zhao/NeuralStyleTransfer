import React, { useReducer, useState, useRef } from 'react';
// import * as mobilenet from '@tensorflow-models/mobilenet';
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

    const contentLayers = [
      'conv_dw_13',
      'conv_pw_13',
    ];
    const styleLayers = [
      'conv_dw_1',
      'conv_pw_1',
      'conv_dw_2',
      'conv_pw_2',
      'conv_dw_3',
      'conv_pw_3',
      'conv_dw_4',
      'conv_pw_4',
      'conv_dw_5',
      'conv_pw_5',
    ];

    // //Test execution 
    // const testStyleExtractor = getFeatureMapExtractor(styleLayers, model);
    // const styleOutputs = testStyleExtractor.predict(htmlImgToTensor(styleImageRef.current));
    // // styleOutputs is of shape (numLayers, featMapheight, FMwidth, FM set depth)

    const styleTargets = await getStyleOutputs(model, styleLayers, htmlImgToTensor(styleImageRef.current));
    const contentTargets = getContentOutputs(model, contentLayers, htmlImgToTensor(contentImageRef.current));
    console.log('got targets');
    console.log(styleTargets);
    console.log(contentTargets);
  }

  // Convert image from any size to a tensor of size
  // [1, 224, 224, 3]
  const htmlImgToTensor = htmlImg => {
    let tensor = tf.browser.fromPixels(htmlImg);
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
  const getStyleOutputs = async (model, styleLayers, styleImgTensor) => {
    const extractor = getFeatureMapExtractor(styleLayers, model);
    const styleFeatMaps = extractor.predict(styleImgTensor);
    // Returns an **array** of tensors
    // Each tensor is an FM of shape (1, featMapheight, FMwidth, FM set depth)
    // FMs are the resulting feature maps from passing a style image into the mobilenet model
    const gramMatrices = await Promise.all(styleFeatMaps.map(ftmap => getGramMatrix(ftmap)));
    return gramMatrices;
  }

  /**
   * Returns an **array** of feature maps of shape (1, height, width, depth)
   * @param {*} model 
   * @param {*} contentLayers 
   * @param {*} contentImgTensor 
   */
  const getContentOutputs = (model, contentLayers, contentImgTensor) => {
    const extractor = getFeatureMapExtractor(contentLayers, model);
    const contentFeatMaps = extractor.predict(contentImgTensor);
    return contentFeatMaps;
  }

  /**
   * Initialize a gram matrix with 0s
   * Gram matrix is of shape (numLayers, FMDepth, FMDepth)
   * @param {*} numLayers 
   * @param {*} depth 
   */
  const getEmptyGramMatrix = (numLayers, depth) => {
    let four_d_array = new Array(numLayers).fill(0)
      .map(() => new Array(depth).fill(0)
        .map(() => new Array(depth).fill(0)));
    return four_d_array;
  }

  /**
   * Shape of inputTensor is (1, 224, 224, depth)
   * @param {*} inputTensor an array of tensors
   * Input tensor shape (numLayers, FMHeight, FMWidth, numFMsInLayer/Depth)
   * Returns an array of gram matrices of FMs for each layer
   */
  const getGramMatrix = async inputTensor => {
    const arr = await inputTensor.array();
    const numLayers = arr.length;
    const height = arr[0].length;
    const width = arr[0][0].length;
    const depth = arr[0][0][0].length;

    let l, i, j, d_a, d_b;
    let retarr = getEmptyGramMatrix(numLayers, depth);
    // Calculate einsum: np.einsum("bijc,bijd->bcd", arr, arr)
    for (l = 0; l < numLayers; l++) {
      for (d_a = 0; d_a < depth; d_a++) {
        for (d_b = 0; d_b < depth; d_b++) {
          for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
              retarr[l][d_a][d_b] += arr[l][i][j][d_a] * arr[l][i][j][d_b];
            }
          }
        }
      }
    }

    // Normalize gram matrix wrt image size
    retarr = retarr.map(layer => layer.map(depth1 => depth1.map(depth2 => depth2 / (height * width))));
    retarr = tf.tensor(retarr);
    return retarr;
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

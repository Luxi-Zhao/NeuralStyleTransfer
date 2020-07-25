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

  // TODO: change this logic to style transfer logic
  // const identify = async () => {
  //   const model = await mobilenet.load({});
  //   const results = await model.classify(contentImageRef.current);
  //   setResults(results);
  // }

  const doTransfer = async () => {
    const model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');

    /**
     * Inspect model 
     */
    // The input size is [null, 224, 224, 3]
    const input_s = model.inputs[0].shape;

    //The output size is [null, 1000]
    const output_s = model.outputs[0].shape;
    console.log(`input size: ${input_s}`);
    console.log(`output size: ${output_s}`);

    //The number of layers in the model '88'
    const len = model.layers.length;
    console.log(`model len: ${len}`);

    //this outputs the name of the 3rd layer 'conv1_relu'
    model.layers.map(layer => {
      console.log(`Layer: ${layer.name}`);
    });
    // ---------- done inspection ---------

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

    //Test execution 
    const testStyleExtractor = getFeatureMapExtractor(styleLayers, model);
    const styleOutputs = testStyleExtractor.predict(htmlImgToTensor(styleImageRef.current));
    console.log(styleOutputs);
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

  const handleReset = () => {
    dispatch('reset');
    setContentImageUrl(null);
    setStyleImageUrl(null);
  }

  const { showLoadingIcon, showResult } = stateMachine.states[state];


  return (
    <div className="App">
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

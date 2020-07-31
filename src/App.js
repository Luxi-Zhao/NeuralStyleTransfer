import React, { useReducer, useState, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Button } from '@material-ui/core';
import CircularProgress from '@material-ui/core/CircularProgress';
import LinearProgressWithLabel from './progressBar';
import './App.css';

import StyleTransferWorker from './styleTransfer.worker.js';

const stateMachine = {
  initial: 'awaitingUpload',
  states: {
    awaitingUpload: { on: { startTransfer: 'transferring' } },
    transferring: { on: { transferDone: 'complete' }, showLoadingIcon: true, showProgressBar: true },
    complete: { on: { reset: 'awaitingUpload' }, showResult: true, showProgressBar: true },
  }
}

const reducer = (currentState, event) => stateMachine.states[currentState].on[event] || stateMachine.initial;

function App() {
  const [state, dispatch] = useReducer(reducer, stateMachine.initial);
  const [contentImageUrl, setContentImageUrl] = useState(null);
  const [styleImageUrl, setStyleImageUrl] = useState(null);
  const [progress, setProgress] = useState(0);
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

    const styleImgArr = await htmlImgToTensor(styleImageRef.current).array();
    const contentImgArr = await htmlImgToTensor(contentImageRef.current).array();

    const styleTransferWorker = new StyleTransferWorker();
    styleTransferWorker.postMessage([styleImgArr, contentImgArr]);
    styleTransferWorker.onmessage = e => {
      const styleTransferProgress = e.data.progress;

      setProgress(styleTransferProgress);
      updateImgPreview(e.data.result, { height: 112, width: 112, depth: 3 }, canvasRef.current);

      if (styleTransferProgress === 100) {
        console.log('transfer done!!')
        dispatch('transferDone');
      }
    }
  }

  function updateImgPreview(imgData, dimensions, canvas) {
    if (canvas == null) return;

    const data = imgData;
    const height = dimensions.height;
    const width = dimensions.width;
    const depth = dimensions.depth;
    console.log(`Height: ${height}, width: ${width}, depth: ${depth}`);
    // console.log(data);
    const multiplier = 255;
    const bytes = new Uint8ClampedArray(width * height * 4);

    for (let i = 0; i < height * width; ++i) {
      let r, g, b, a;
      r = data[i * 3] * multiplier;
      g = data[i * 3 + 1] * multiplier;
      b = data[i * 3 + 2] * multiplier;
      a = 255;

      const j = i * 4;
      bytes[j + 0] = Math.round(r);
      bytes[j + 1] = Math.round(g);
      bytes[j + 2] = Math.round(b);
      bytes[j + 3] = Math.round(a);
    }

    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const imageData = new ImageData(bytes, width, height);
    ctx.putImageData(imageData, 0, 0);
  }

  // Convert image from any size to a tensor of size
  // [1, 112, 112, 3]
  // TODO deal with rectangular images
  const htmlImgToTensor = htmlImg => {
    let tensor = tf.browser.fromPixels(htmlImg);
    tensor = tf.div(tensor, 255.0);
    tensor = tf.image.resizeBilinear(tensor, [112, 112]);
    tensor = tf.expandDims(tensor, 0);
    return tensor;
  }

  const handleReset = () => {
    dispatch('reset');
    setContentImageUrl(null);
    setStyleImageUrl(null);
    setProgress(0);
  }

  const { showLoadingIcon, showProgressBar, showResult } = stateMachine.states[state];


  return (
    <div className="App">
      {<canvas
        id="canvas"
        ref={canvasRef}
        width={112}
        height={112}
        style={{
          border: '2px solid #000',
          marginTop: 10,
        }}
      />}
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
      {showProgressBar && <div>
        <LinearProgressWithLabel value={progress} />
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

import React, { useReducer, useState, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Button } from '@material-ui/core';
import CircularProgress from '@material-ui/core/CircularProgress';
import LinearProgressWithLabel from './progressBar';
import './App.css';

import Worker from './styleTransfer.worker.js';

const stateMachine = {
  initial: 'awaitingUpload',
  states: {
    awaitingUpload: { on: { startTransfer: 'transferring' } },
    transferring: { on: { transferDone: 'complete' }, showLoadingIcon: true },
    complete: { on: { reset: 'awaitingUpload' }, showResult: true },
  }
}

const reducer = (currentState, event) => stateMachine.states[currentState].on[event] || stateMachine.initial;

function App() {
  const [state, dispatch] = useReducer(reducer, stateMachine.initial);
  const [contentImageUrl, setContentImageUrl] = useState(null);
  const [styleImageUrl, setStyleImageUrl] = useState(null);
  const [progress, setProgress] = useState(0);
  const [worker, setWorker] = useState(null);
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

    const styleImgArr = htmlImgToTensor(styleImageRef.current).arraySync();
    const contentImgArr = htmlImgToTensor(contentImageRef.current).arraySync();

    const worker = new Worker();
    setWorker(worker);
    worker.postMessage([styleImgArr, contentImgArr]);

  }

  if (worker) {
    worker.onmessage = e => {
      const styleTransferProgress = e.data.progress;
      const result = e.data.result;

      // How to incrementally set progress and updates? 
      setProgress(styleTransferProgress);
      console.log(`Setting progress: ${styleTransferProgress}`);
      tf.browser.toPixels(tf.tensor(result), canvasRef.current);

      if (styleTransferProgress === 100) {
        console.log('transfer done!!')
        dispatch('transferDone');
      }
    }
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

  const { showLoadingIcon, showResult } = stateMachine.states[state];


  return (
    <div className="App">
      {<canvas
        id="canvas"
        ref={canvasRef}
        width={500}
        height={500}
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

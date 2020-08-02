import React, { useReducer, useState, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Button, CircularProgress, Grid, Card, Box, Divider, Slide, Typography } from '@material-ui/core';
import LinearProgressWithLabel from './progressBar';
import './App.css';

import StyleTransferWorker from './styleTransfer.worker.js';

const stateMachine = {
  initial: 'awaitingUpload',
  states: {
    awaitingUpload: { on: { startTransfer: 'transferring' } },
    transferring: { on: { transferDone: 'complete' }, showLoadingIcon: true, showResultSection: true },
    complete: { on: { reset: 'awaitingUpload' }, showReset: true, showResultSection: true },
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
      // Reset file input so that onChange is always triggered
      e.target.value = null;
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

  /**
   * Adapted from 
   * https://github.com/tensorflow/tfjs-core/blob/v1.0.0/src/ops/browser.ts#L73-L159
   * @param {*} imgData 
   * @param {*} dimensions 
   * @param {*} canvas 
   */
  function updateImgPreview(imgData, dimensions, canvas) {
    if (canvas == null) return;

    const data = imgData;
    const height = dimensions.height;
    const width = dimensions.width;
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

  const { showLoadingIcon, showResultSection, showReset } = stateMachine.states[state];


  const cardStyleLeft = {
    marginTop: '5vw',
  }
  const cardStyleRight = {
    marginTop: '5vw',
  }
  const dividerStyle = {
    marginLeft: 'auto',
    marginRight: 'auto',
    height: '15vw',
  }

  return (
    <div className="App">
      <Grid container spacing={7} direction="column" id="root-container">
        {<Grid item id="section-upload">
          <Grid container spacing={0} justify={'center'} id="section-upload-img">
            <Grid item xs={5} id="section-upload-img-content">
              <Card style={cardStyleLeft}>
                <Box
                  display="flex"
                  alignItems="center"
                  p={1}
                  css={{ height: '30vw' }}>
                  {<img alt="" src={contentImageUrl} ref={contentImageRef} />}
                </Box>
              </Card>
            </Grid>
            <Grid item xs={1} id="section-upload-btn-combine">
              <Box marginTop={"5vw"}>
                <Divider orientation="vertical" style={dividerStyle} />
                <Button
                  color="primary"
                  aria-label="outlined primary"
                  onClick={handleStyleTransfer}
                  disabled={!(contentImageUrl && styleImageUrl)}
                >
                  Combine
                </Button>
                <Divider orientation="vertical" style={dividerStyle} />
              </Box>
            </Grid>
            <Grid item xs={5} id="section-upload-img-style">
              <Card style={cardStyleRight}>
                <Box
                  display="flex"
                  alignItems="center"
                  p={1}
                  css={{ height: '30vw' }}>
                  {<img alt="" src={styleImageUrl} ref={styleImageRef} />}
                </Box>
              </Card>
            </Grid>
          </Grid>
          <Grid container spacing={3} id="section-upload-btns">
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
            <Grid item xs id="section-upload-btn-content">
              <label htmlFor="content-file-input">
                <Button
                  variant="outlined"
                  color="primary"
                  aria-label="outlined primary"
                  onClick={() => contentInputRef.current.click()}
                >
                  Upload Content Image
              </Button>
              </label>
            </Grid>
            <Grid item xs id="section-upload-btn-style">
              <label htmlFor="style-file-input">
                <Button
                  variant="outlined"
                  color="primary"
                  aria-label="outlined primary"
                  onClick={() => styleInputRef.current.click()}
                >
                  Upload Style Image
              </Button>
              </label>
            </Grid>
          </Grid>
        </Grid>}
        <Grid item id="section-result">
          <Slide direction="up" in={showResultSection} mountOnEnter unmountOnExit>
            <Box boxShadow={4} p={"5vw"}>
              <Grid
                container
                direction="column"
                spacing={5}
                justify="center"
                alignItems="center">
                <Grid item>
                  <Typography variant="h4" gutterBottom color="primary">
                    Result Image
                  </Typography>
                  <Card marginTop="2vw">
                    <canvas
                      id="canvas"
                      ref={canvasRef}
                      width={112}
                      height={112}
                    />
                  </Card>
                </Grid>
                <Grid item>
                  {showLoadingIcon &&
                    <CircularProgress />
                  }
                </Grid>
                <Grid item>
                  {<Box width="30vw">
                    <LinearProgressWithLabel value={progress} />
                  </Box>}
                </Grid>
                <Grid item>
                  {showReset &&
                    <Button
                      variant="outlined"
                      color="primary"
                      aria-label="outlined primary"
                      onClick={handleReset}
                    >
                      Reset
                  </Button>}
                </Grid>
              </Grid>
            </Box>
          </Slide>
        </Grid>
      </Grid>
    </div>
  );
}

export default App;

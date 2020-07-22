import React, { useReducer, useState, useRef } from 'react';
import * as mobilenet from '@tensorflow-models/mobilenet';
import './App.css';

const stateMachine = {
  initial: 'initial',
  states: {
    initial: { on: { next: 'loadingModel' } },
    loadingModel: { on: { next: 'awaitingUpload' } },
    awaitingUpload: { on: { next: 'ready' } },
    ready: { on: { next: 'classifying' }, showImage: true },
    classifying: { on: { next: 'complete' } },
    complete: { on: { next: 'awaitingUpload' }, showImage: true, showResults: true },
  }
}

// is the event always next? - yes it is
const reducer = (currentState, event) => stateMachine.states[currentState].on[event] || stateMachine.initial;

const formatResult = ({ className, probability }) => (
  <li key={className}>
    {`${className}: %${(probability * 100).toFixed(2)}`}
  </li>
)


function App() {
  // when a dispatch action fires, the page re-renders, App() executes, and state gets
  // updated
  const [state, dispatch] = useReducer(reducer, stateMachine.initial);
  const [model, setModel] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [results, setResults] = useState([]);
  const inputRef = useRef();
  const imageRef = useRef();

  const next = () => dispatch('next');

  const loadModel = async () => {
    next(); // transition

    const mobilenetModel = await mobilenet.load();
    setModel(mobilenetModel);

    next(); // transition
  }

  const handleUpload = e => {
    const { files } = e.target;
    if (files.length > 0) {
      const url = URL.createObjectURL(files[0]);
      setImageUrl(url);

      next();
    }
  }

  const identify = async () => {
    next();
    const results = await model.classify(imageRef.current);
    setResults(results);
    next();
  }

  const buttonProps = {
    initial: { text: 'Load Model', action: loadModel },
    loadingModel: { text: 'Loading Model...', action: () => { } },
    awaitingUpload: { text: 'Upload Photo', action: () => inputRef.current.click() },
    ready: { text: 'Identify', action: identify },
    classifying: { text: 'Identifying', action: () => { } },
    complete: { text: 'Reset', action: next },
  }

  const { showImage, showResults } = stateMachine.states[state]; // showImage takes on the value of the key

  return (
    <div className="App">
      {showImage && <img alt="upload-preview" src={imageUrl} ref={imageRef} />}
      {showResults && <ul>
        {results.map(formatResult)}
      </ul>}
      <input type="file" accept="image/*" ref={inputRef} onChange={handleUpload} />
      <button onClick={buttonProps[state].action}>{buttonProps[state].text}</button>
    </div>
  );
}

export default App;

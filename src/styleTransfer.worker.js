import { doTransfer } from './styleTransfer';

onmessage = async e => {
    console.log('Worker: Message received from app.js');
    const styleImgArr = e.data[0];
    const contentImgArr = e.data[1];

    const onEpochDone = (progress, intermediaryResult) => {
        postMessage({
            progress,
            result: intermediaryResult,
        });
    };

    await doTransfer(styleImgArr, contentImgArr, onEpochDone);
}
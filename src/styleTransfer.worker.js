import { doTransfer } from './styleTransfer';

onmessage = async e => {
    console.log('Worker: Message received from app.js');
    const styleImgArr = e.data[0];
    const contentImgArr = e.data[1];

    const result = await doTransfer(styleImgArr, contentImgArr);
    // result should be a [height, width, depth] array
    console.log('Worker: Posting message back to main script');
    postMessage(result);
}
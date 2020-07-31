import * as tf from '@tensorflow/tfjs';

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

const doTransfer = async (styleImgArr, contentImgArr, onEpochDone) => {
    console.log('In do transfer')

    const model = await tf.loadLayersModel('http://localhost:8080/model.json');

    const styleImgTensor = tf.tensor(styleImgArr);
    const contentImgTensor = tf.tensor(contentImgArr);

    const styleExtractor = getFeatureMapExtractor(styleLayers, model);
    const contentExtractor = getFeatureMapExtractor(contentLayers, model);

    const styleTargets = getStyleOutputs(styleExtractor, styleImgTensor);
    const contentTargets = getContentOutputs(contentExtractor, contentImgTensor);

    console.log('got style and content targets. Starting to graddesc.');

    const optimizer = tf.train.adam(0.02, 0.99, undefined, 0.1);
    const resultImage = tf.variable(contentImgTensor);

    const epochs = 10;
    const stepsPerEpoch = 1;
    let epoch, step;
    for (epoch = 0; epoch < epochs; epoch++) {
        for (step = 0; step < stepsPerEpoch; step++) {
            trainStep(
                resultImage, optimizer, styleTargets, contentTargets, styleExtractor, contentExtractor);
            console.log('.');
        }

        // Call callback at the end of each epoch
        const progress = (epoch + 1) / epochs * 100;
        const intermediaryResult = await resultImage.squeeze().data();
        onEpochDone(progress, intermediaryResult);

        console.log(`Epoch #${epoch} done.`)
    }
}

const trainStep = (image, optimizer, styleTargets, contentTargets, styleExtractor, contentExtractor) => {
    const lossFunction = () => tf.tidy(() => {
        const styleOutputs = getStyleOutputs(styleExtractor, image);
        const contentOutputs = getContentOutputs(contentExtractor, image);

        const styleWeight = 0.01;
        const contentWeight = 1000;
        const totalVariationWeight = 10000;

        let styleLoss = tf.addN(styleOutputs.map((styleOutput, index) => tf.mean(
            tf.square(tf.sub(styleOutput, styleTargets[index]))
        )));
        styleLoss = tf.mul(tf.div(styleLoss, styleLayers.length), styleWeight);

        let contentLoss = tf.addN(contentOutputs.map((contentOutput, index) => tf.mean(
            tf.square(tf.sub(contentOutput, contentTargets[index]))
        )));
        contentLoss = tf.mul(tf.div(contentLoss, contentLayers.length), contentWeight);

        const variationLoss = tf.mul(totalVariationLoss(image), totalVariationWeight);
        const loss = tf.add(styleLoss, contentLoss, variationLoss);
        return loss;
    });

    optimizer.minimize(lossFunction, false, [image]);
    image.assign(clip_0_1(image));
}

/**
 * @param {*} image a tensor
 */
const clip_0_1 = image => {
    return tf.clipByValue(image, 0, 1);
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

const highPassXY = t => {
    const height = t.shape[1];
    const width = t.shape[2];
    const xDelta = tf.sub(tf.slice4d(t, [0, 0, 1, 0], [-1, -1, -1, -1]), tf.slice4d(t, [0, 0, 0, 0], [-1, -1, width - 1, -1]));
    const yDelta = tf.sub(tf.slice4d(t, [0, 1, 0, 0], [-1, -1, -1, -1]), tf.slice4d(t, [0, 0, 0, 0], [-1, height - 1, -1, -1]));
    return { xDelta, yDelta };
}

const totalVariationLoss = t => {
    const { xDelta, yDelta } = highPassXY(t);
    return tf.add(tf.sum(tf.abs(xDelta)), tf.sum(tf.abs(yDelta)));
}

export { doTransfer };

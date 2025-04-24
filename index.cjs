const tf = require('@tensorflow/tfjs');

const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

// Prepare the model for training: Specify the loss and the optimizer.
model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

(async () => {
    // loop for training it 1000 times
    let arr1 = [];
    let arr2 = [];

    for (let i = 0; i < 1000000000; i++) {
        arr1.push(i);
        arr2.push(i * 2);
    }

    const xs = tf.tensor2d(arr1, [1000000000, 1]);
    const ys = tf.tensor2d(arr2, [1000000000, 1]);

    // Train the model using the data.
    await model.fit(xs, ys, { epochs: 250 });
})();

console.log(model.predict(tf.tensor2d([1], [1, 1])).dataSync());
import tf from '@tensorflow/tfjs';
import fs from 'fs';

// const defaults = JSON.parse(fs.readFileSync('defaults.json'));

const alphToNum = (str) => {
    const alphabet = 'abcdefghijklmnopqrstuvwxyz ';
    const num = str.split('').map((char) => {
        const index = alphabet.indexOf(char.toLowerCase());
        return index !== -1 ? index : alphabet.length; // Return 0 for non-alphabetic characters
    });

    return num;
};

const numToAlph = (num) => {
    const alphabet = 'abcdefghijklmnopqrstuvwxyz ';
    const str = num.map((index) => {
        return index < alphabet.length ? alphabet[index] : '?'; // Return '?' for out-of-bounds indices
    });
    return str.join('');
}

const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [3] }));

// Prepare the model for training: Specify the loss and the optimizer.
model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

// let arr1 = [];
// let arr2 = [];

// // loop for training it 1000 times
// for (let i = 0; i < 100; i++) {
//     arr1.push(i);
//     arr2.push(i * 2);
// }

const xs = tf.tensor2d([
    [2, 1, 3],   // 2 + 3 = 5
    [10, 2, 4],  // 10 - 4 = 6
    [6, 3, 2],   // 6 * 2 = 12
    [12, 4, 3],  // 12 / 3 = 4

    [0, 1, 5],   // 0 + 5 = 5
    [7, 2, 10],  // 7 - 10 = -3
    [4, 3, 4],   // 4 * 4 = 16
    [16, 4, 2],  // 16 / 2 = 8

    [5, 1, 0],   // 5 + 0 = 5
    [0, 2, 5],   // 0 - 5 = -5
    [3, 3, 3],   // 3 * 3 = 9
    [9, 4, 3],   // 9 / 3 = 3

    [1, 1, 9],   // 1 + 9 = 10
    [15, 2, 7],  // 15 - 7 = 8
    [8, 3, 1],   // 8 * 1 = 8
    [20, 4, 5],  // 20 / 5 = 4

    [6, 1, 7],   // 6 + 7 = 13
    [3, 2, 8],   // 3 - 8 = -5
    [2, 3, 9],   // 2 * 9 = 18
    [18, 4, 6]   // 18 / 6 = 3]
], [20, 3]);

const ys = tf.tensor1d([
    5, 6, 12, 4,
    5, -3, 16, 8,
    5, -5, 9, 3,
    10, 8, 8, 4,
    13, -5, 18, 3
]);

// Train the model using the data.
model.fit(xs, ys, { epochs: 350 }).then(() => {
    console.log(model.predict(tf.tensor2d([[4, 3, 2]], [1, 3])).dataSync());

    model.save('file://./defaults').then(() => {
        console.log('Model saved to defaults.json');
    }).catch((err) => {
        console.error('Error saving model:', err);
    });
});

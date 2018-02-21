/**
 * 预测线性方程 y = 3x + 2 的参数
 * 其中 3 为 Weights, 2 为 Biases 
 */

// 创建数据集，这里创建5个x，和5个y
const x_data = dl.tensor1d([0, 1, 2, 3, 4]);
const a = dl.scalar(3)
const b = dl.scalar(2)
const y_data = x_data.mul(a).add(b)

// 接着生成我们要求解的两个参数Weights和Biases
const Weights = dl.variable(dl.randomUniform([1]))
const Biases = dl.variable(dl.zeros([1]))

// 接着定义预测的y值，损失函数和optimizer
// 损失函数指的是预测值与实际值之间的差别
// 神经网络的重点就是通过优化器来减少误差，提升参数的准确度
// 这里用平方差值作为损失函数，用gradient descent作为优化器
const f = x => Weights.mul(x).add(Biases);
const loss = (pred, label) => pred.sub(label).square().mean()
const learningRate = 0.01
const optimizer = dl.train.sgd(learningRate)

// 训练模型，训练500次
for (let i = 0; i < 500; i++) {
    optimizer.minimize(() => loss(f(x_data), y_data))
}

// 预测出参数Weights和Biases
console.log(
    `Weights: ${Weights.dataSync()}, Biases: ${Biases.dataSync()}`
)
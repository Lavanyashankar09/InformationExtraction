var maryMarkov = require("mary-markov")

let hiddenStates = [    
    {state: 'Sunny', prob: [0.8, 0.2]},    
    {state: 'Rainy', prob: [0.4, 0.6]}     
];

let observables = [    
    {obs: 'Paint', prob: [0.4, 0.3]},    
    {obs: 'Clean', prob: [0.1, 0.45]},    
    {obs: 'Shop', prob: [0.2, 0.2]},    
    {obs: 'Bike', prob: [0.3, 0.05]}    
];

let hiddenInit = [0.6, 0.4];

let LeaHMModel = maryMarkov.HMM(hiddenStates, observables, hiddenInit);

let obSequence = ['Paint','Clean','Shop','Bike']; 

let forwardProbability = LeaHMModel.forwardAlgorithm(obSequence);
console.log(forwardProbability.alphaF);
//0.0031560000000000004

let backwardProbability = LeaHMModel.backwardAlgorithm(obSequence);
console.log(backwardProbability.betaF); 
//0.003156000000000001
// import * as tf from '@tensorflow/tfjs';
import Webcam from './webcam';

const webcam = new Webcam(document.getElementById('webcamElement'));

const img = webcam.capture();
var display = document.createElement("display");
// console.log()
display.setAttribute("src", img);
display.setAttribute("style", "float:right");
console.log("hello?");
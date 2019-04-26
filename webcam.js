


"use strict";
export class Webcam {
    /**
     * @param {HTMLVideoElement} webcamElement A HTMLVideoElement representing the webcam feed.
     */
    constructor(webcamElement) {
      this.webcamElement = webcamElement;
    }
    
    /**
     * Captures a frame from the webcam and normalizes it between -1 and 1.
     * Returns a batched image (1-element batch) of shape [1, w, h, c].
     */
    capture() {
      video = document.querySelector("#webcamElement");
      canvas = document.getElementById("container"); //if this doesn't work, try videoElement
      ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    }
  
    /**
     * Crops an image tensor so we get a square image with no white space.
     * @param {Tensor4D} img An input image Tensor to crop.
     */
    cropImage(img) {
      const size = Math.min(img.shape[0], img.shape[1]);
      const centerHeight = img.shape[0] / 2;
      const beginHeight = centerHeight - (size / 2);
      const centerWidth = img.shape[1] / 2;
      const beginWidth = centerWidth - (size / 2);
      return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
    }
  }
//   exports.Webcam = Webcam;
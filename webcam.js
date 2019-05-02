


// "use strict";
export default class Webcam {
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
    // capture() {
    //   const video = document.querySelector("#webcamElement");
    //   var canvas = document.getElementById("canvas"); //if this doesn't work, try videoElement (previously container)
    //   var ctx = canvas.getContext('2d');
    //   video.addEventListener('play', draw(video, canvas, ctx, 3));
    // }

    // draw(video, canvas, context, frameRate) {
    //     ret = ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    //     setTimeout(draw, 1/frameRate, video, canvas, context, frameRate);
    //     return ret;
    // }
    
  
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
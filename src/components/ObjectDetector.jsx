import React, { useRef, useState } from "react";
import "@tensorflow/tfjs-backend-cpu";
import * as cocoSsd from "@tensorflow-models/coco-ssd";

export function ObjectDetector(props) {
  const fileInputRef = useRef();
  const imageRef = useRef();
  const [imgData, setImgData] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [isLoading, setLoading] = useState(false);

  const isEmptyPredictions = !predictions || predictions.length === 0;

  const openFilePicker = () => {
    if (fileInputRef.current) fileInputRef.current.click();
  };

  const normalizePredictions = (predictions, imgSize) => {
    if (!predictions || !imgSize || !imageRef) return predictions || [];
    return predictions.map((prediction) => {
      const { bbox } = prediction;
      const oldX = bbox[0];
      const oldY = bbox[1];
      const oldWidth = bbox[2];
      const oldHeight = bbox[3];

      const imgWidth = imageRef.current.width;
      const imgHeight = imageRef.current.height;

      const x = (oldX * imgWidth) / imgSize.width;
      const y = (oldY * imgHeight) / imgSize.height;
      const width = (oldWidth * imgWidth) / imgSize.width;
      const height = (oldHeight * imgHeight) / imgSize.height;

      return { ...prediction, bbox: [x, y, width, height] };
    });
  };

  const detectObjectsOnImage = async (imageElement, imgSize) => {
    const model = await cocoSsd.load({});
    const predictions = await model.detect(imageElement, 6);
    const normalizedPredictions = normalizePredictions(predictions, imgSize);
    setPredictions(normalizedPredictions);
  };

  const readImage = (file) => {
    return new Promise((rs, rj) => {
      const fileReader = new FileReader();
      fileReader.onload = () => rs(fileReader.result);
      fileReader.onerror = () => rj(fileReader.error);
      fileReader.readAsDataURL(file);
    });
  };

  const onSelectImage = async (e) => {
    setPredictions([]);
    setLoading(true);

    const file = e.target.files[0];
    const imgData = await readImage(file);
    setImgData(imgData);

    const imageElement = document.createElement("img");
    imageElement.src = imgData;

    imageElement.onload = async () => {
      const imgSize = {
        width: imageElement.width,
        height: imageElement.height,
      };
      await detectObjectsOnImage(imageElement, imgSize);
      setLoading(false);
    };
  };

  return (
    <div className="object-container">
      <div className="detector-container">
        {imgData && <img src={imgData} ref={imageRef} alt="img" />}
        {!isEmptyPredictions &&
          predictions.map((prediction, idx) => (
            <div className="data" key={idx}>
              <span>{`${prediction.class} ${(prediction.score * 100).toFixed(
                1
              )}%`}</span>
              {document.querySelector(".data") &&
                document
                  .querySelector(".data")
                  .style.setProperty("left", `${prediction.bbox[0]}px`)}
              {document.querySelector(".data") &&
                document
                  .querySelector(".data")
                  .style.setProperty("top", `${prediction.bbox[1]}px`)}
              {document.querySelector(".data") &&
                document
                  .querySelector(".data")
                  .style.setProperty("width", `${prediction.bbox[2]}px`)}
              {document.querySelector(".data") &&
                document
                  .querySelector(".data")
                  .style.setProperty("height", `${prediction.bbox[3]}px`)}
            </div>
          ))}
      </div>
      <input type="file" ref={fileInputRef} onChange={onSelectImage} />
      <button onClick={openFilePicker}>
        {isLoading ? "Recognizing..." : "Select Image"}
      </button>
    </div>
  );
}

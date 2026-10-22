const API_BASE_URL = window.PLANT_API_BASE_URL || "http://127.0.0.1:8000";
const API_URL = new URL("/api/v1/predict", API_BASE_URL).toString();
const MAX_FILE_BYTES = 10 * 1024 * 1024;

const imageInput = document.querySelector("#imageInput");
const previewImage = document.querySelector("#previewImage");
const cameraVideo = document.querySelector("#cameraVideo");
const captureCanvas = document.querySelector("#captureCanvas");
const emptyPreview = document.querySelector("#emptyPreview");
const previewTag = document.querySelector("#previewTag");
const photoActions = document.querySelector("#photoActions");
const cameraActions = document.querySelector("#cameraActions");
const retakeActions = document.querySelector("#retakeActions");
const startCameraButton = document.querySelector("#startCameraButton");
const captureButton = document.querySelector("#captureButton");
const stopCameraButton = document.querySelector("#stopCameraButton");
const predictButton = document.querySelector("#predictButton");
const retakeButton = document.querySelector("#retakeButton");
const chooseAnotherButton = document.querySelector("#chooseAnotherButton");
const resultEmpty = document.querySelector("#resultEmpty");
const resultContent = document.querySelector("#resultContent");
const loadingState = document.querySelector("#loadingState");
const errorMessage = document.querySelector("#errorMessage");
const cameraHint = document.querySelector("#cameraHint");
const predictedClass = document.querySelector("#predictedClass");
const confidenceValue = document.querySelector("#confidenceValue");
const confidenceBar = document.querySelector("#confidenceBar");
const otherPredictions = document.querySelector("#otherPredictions");

let selectedFile = null;
let cameraStream = null;
let previewUrl = null;

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

function clearError() {
  errorMessage.textContent = "";
  errorMessage.hidden = true;
}

function clearResult() {
  resultContent.hidden = true;
  resultEmpty.hidden = false;
  loadingState.hidden = true;
  clearError();
}

function setPreview(source) {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = source;
  previewImage.src = source;
  previewImage.hidden = false;
  cameraVideo.hidden = true;
  emptyPreview.hidden = true;
  previewTag.hidden = false;
  cameraHint.textContent = "";
}

function releaseCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
  cameraVideo.srcObject = null;
  cameraVideo.hidden = true;
}

function showPreviewActions() {
  photoActions.hidden = true;
  cameraActions.hidden = true;
  retakeActions.hidden = false;
}

function resetToPicker() {
  releaseCamera();
  selectedFile = null;
  imageInput.value = "";
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
  previewImage.removeAttribute("src");
  previewImage.hidden = true;
  previewTag.hidden = true;
  emptyPreview.hidden = false;
  photoActions.hidden = false;
  cameraActions.hidden = true;
  retakeActions.hidden = true;
  cameraHint.textContent = "Camera access works on localhost or HTTPS. You can also upload a photo.";
  clearResult();
}

async function predictImage(file) {
  loadingState.hidden = false;
  resultEmpty.hidden = true;
  resultContent.hidden = true;
  clearError();

  const formData = new FormData();
  formData.append("file", file, file.name || "leaf-photo.jpg");
  try {
    const response = await fetch(API_URL, { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "The image could not be analyzed.");
    }
    renderResult(payload);
  } catch (error) {
    resultEmpty.hidden = true;
    showError(error instanceof TypeError
      ? "Couldn’t reach the prediction API. Start the backend and try again."
      : error.message);
  } finally {
    loadingState.hidden = true;
  }
}

function renderResult(result) {
  predictedClass.textContent = formatClassName(result.predicted_class);
  confidenceValue.textContent = `${result.confidence.toFixed(2)}%`;
  confidenceBar.style.width = `${Math.max(0, Math.min(100, result.confidence))}%`;
  otherPredictions.replaceChildren();

  result.top_predictions.slice(1).forEach((prediction) => {
    const row = document.createElement("div");
    row.className = "prediction-row";
    const label = document.createElement("span");
    label.textContent = formatClassName(prediction.class_name);
    const score = document.createElement("span");
    score.textContent = `${prediction.confidence.toFixed(2)}%`;
    row.append(label, score);
    otherPredictions.append(row);
  });

  resultEmpty.hidden = true;
  resultContent.hidden = false;
}

function formatClassName(className) {
  return className.replaceAll("_", " ").replace(/\s+/g, " ").trim();
}

function acceptImage(file) {
  if (!file.type.startsWith("image/")) {
    showError("Choose an image file such as JPG, PNG, or WebP.");
    return;
  }
  if (file.size > MAX_FILE_BYTES) {
    showError("Choose an image smaller than 10 MB.");
    return;
  }
  releaseCamera();
  selectedFile = file;
  setPreview(URL.createObjectURL(file));
  showPreviewActions();
  clearResult();
}

imageInput.addEventListener("change", () => {
  const file = imageInput.files?.[0];
  if (file) acceptImage(file);
});

async function startCamera() {
  clearError();
  if (!navigator.mediaDevices?.getUserMedia) {
    showError("Camera access needs localhost or HTTPS. You can upload a photo instead.");
    return;
  }
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } },
      audio: false,
    });
    cameraVideo.srcObject = cameraStream;
    cameraVideo.hidden = false;
    previewImage.hidden = true;
    emptyPreview.hidden = true;
    previewTag.hidden = true;
    photoActions.hidden = true;
    cameraActions.hidden = false;
    retakeActions.hidden = true;
    cameraHint.textContent = "Camera is on. Frame one clear leaf, then capture it.";
    clearResult();
  } catch (error) {
    showError(error.name === "NotAllowedError"
      ? "Camera permission was denied. Allow camera access in browser settings or upload a photo."
      : "Couldn’t start the camera. Check that it is connected and not being used by another app.");
  }
}

startCameraButton.addEventListener("click", startCamera);

captureButton.addEventListener("click", () => {
  if (!cameraVideo.videoWidth || !cameraVideo.videoHeight) {
    showError("The camera is still starting. Try again in a moment.");
    return;
  }
  captureCanvas.width = cameraVideo.videoWidth;
  captureCanvas.height = cameraVideo.videoHeight;
  const context = captureCanvas.getContext("2d");
  context.drawImage(cameraVideo, 0, 0);
  captureCanvas.toBlob((blob) => {
    if (!blob) {
      showError("Couldn’t capture the photo. Please try again.");
      return;
    }
    const file = new File([blob], "webcam-capture.jpg", { type: "image/jpeg" });
    releaseCamera();
    selectedFile = file;
    setPreview(URL.createObjectURL(file));
    showPreviewActions();
    clearResult();
    predictImage(file);
  }, "image/jpeg", 0.92);
});

stopCameraButton.addEventListener("click", resetToPicker);
predictButton.addEventListener("click", () => {
  if (selectedFile) predictImage(selectedFile);
});
retakeButton.addEventListener("click", async () => {
  if (!navigator.mediaDevices?.getUserMedia) {
    showError("Camera access needs localhost or HTTPS. Choose another photo instead.");
    return;
  }
  await startCamera();
});
chooseAnotherButton.addEventListener("click", resetToPicker);
window.addEventListener("pagehide", releaseCamera);

const form = document.querySelector("#transaction-form");
const fields = [...form.querySelectorAll("input")];
const pcaFields = fields.filter((field) => /^V\d+$/.test(field.name));
const predictButton = document.querySelector("#predict-button");
const formError = document.querySelector("#form-error");
const formStatus = document.querySelector("#form-status");
const resultCard = document.querySelector("#result-card");
const emptyResult = document.querySelector("#empty-result");
const predictionResult = document.querySelector("#prediction-result");
const exampleButtons = document.querySelector("#example-buttons");
let modelReady = false;
let requestVersion = 0;
let activePrediction = null;

async function getJSON(url, options = {}) {
  const response = await fetch(url, options);
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(
      "The server returned an unreadable response. Please try again.",
    );
  }
  if (!response.ok) {
    const error = new Error(
      data.error || "The request could not be completed. Please try again.",
    );
    error.fields = data.fields || [];
    throw error;
  }
  return data;
}

function setBusy(busy) {
  predictButton.disabled = busy || !modelReady;
  document.querySelector("#predict-button-text").textContent = busy
    ? "Analyzing…"
    : "Analyze transaction";
  form.setAttribute("aria-busy", String(busy));
  resultCard.setAttribute("aria-busy", String(busy));
}

function showError(message, fieldNames = []) {
  formError.textContent = message;
  formError.hidden = false;
  fields.forEach((field) =>
    field.setAttribute("aria-invalid", String(fieldNames.includes(field.name))),
  );
  const firstInvalid = fields.find((field) => fieldNames.includes(field.name));
  if (firstInvalid) {
    if (pcaFields.includes(firstInvalid))
      document.querySelector("#pca-section").open = true;
    firstInvalid.focus();
  }
}

function updateCompleteness() {
  const completePca = pcaFields.filter(
    (field) => field.value !== "" && field.validity.valid,
  ).length;
  const count = document.querySelector("#feature-count");
  count.textContent = `${completePca} of 28 entered`;
  count.classList.toggle("complete", completePca === 28);
  const complete = fields.filter(
    (field) => field.value !== "" && field.validity.valid,
  ).length;
  formStatus.textContent =
    complete === 30
      ? "All inputs ready for analysis"
      : `${complete} of 30 inputs entered`;
}

function invalidatePrediction() {
  requestVersion += 1;
  if (activePrediction) activePrediction.abort();
  activePrediction = null;
  setBusy(false);
  formError.hidden = true;
  fields.forEach((field) => field.removeAttribute("aria-invalid"));
  resultCard.classList.remove("legitimate", "fraudulent");
  predictionResult.hidden = true;
  emptyResult.hidden = false;
  document.querySelector("#empty-title").textContent = "Ready when you are";
  document.querySelector("#empty-copy").textContent =
    "Complete the transaction details and select Analyze transaction to see the model's prediction.";
  document.querySelector("#result-state").textContent = "Awaiting data";
}

function renderPrediction(result) {
  const score = result.fraud_probability * 100;
  const fraudulent = result.prediction === 1;
  resultCard.classList.toggle("fraudulent", fraudulent);
  resultCard.classList.toggle("legitimate", !fraudulent);
  document.querySelector("#prediction-label").textContent = result.label;
  document.querySelector("#outcome-icon").textContent = fraudulent ? "!" : "✓";
  document.querySelector("#prediction-description").textContent = fraudulent
    ? "This transaction meets the model's fraud threshold."
    : "This transaction falls below the model's fraud threshold.";
  document.querySelector("#fraud-score").textContent = `${score.toFixed(2)}%`;
  document.querySelector("#score-fill").style.width = `${score}%`;
  document
    .querySelector("#score-meter")
    .setAttribute("aria-valuenow", score.toFixed(2));
  document.querySelector("#threshold-marker").style.left =
    `${result.threshold * 100}%`;
  document.querySelector("#score-threshold").textContent =
    `${result.threshold * 100}% threshold`;
  document.querySelector("#result-amount").textContent =
    result.transaction.Amount.toLocaleString(undefined, {
      maximumFractionDigits: 8,
    });
  document.querySelector("#result-time").textContent =
    `${result.transaction.Time.toLocaleString(undefined, { maximumFractionDigits: 8 })} s`;
  document.querySelector("#result-hours").textContent = result.features.Hours;
  document.querySelector("#result-logamount").textContent =
    result.features.LogAmount.toFixed(5);
  document.querySelector("#result-state").textContent = "Analysis complete";
  emptyResult.hidden = true;
  predictionResult.hidden = false;
}

form.addEventListener("input", () => {
  invalidatePrediction();
  updateCompleteness();
  document.querySelector("#example-message").hidden = true;
});

document.querySelector("#reset-button").addEventListener("click", () => {
  form.reset();
  invalidatePrediction();
  updateCompleteness();
  document.querySelector("#example-message").hidden = true;
  document.querySelector("#Amount").focus();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!modelReady || activePrediction) return;
  formError.hidden = true;
  const invalidFields = fields.filter(
    (field) => !field.validity.valid || !Number.isFinite(Number(field.value)),
  );
  if (invalidFields.length) {
    showError(
      "Enter a valid number in every field. Amount and elapsed time must be zero or greater.",
      invalidFields.map((field) => field.name),
    );
    return;
  }

  const transaction = Object.fromEntries(
    fields.map((field) => [field.name, Number(field.value)]),
  );
  const version = ++requestVersion;
  const controller = new AbortController();
  activePrediction = controller;
  const timeout = setTimeout(() => controller.abort(), 30000);
  setBusy(true);
  resultCard.classList.remove("legitimate", "fraudulent");
  predictionResult.hidden = true;
  emptyResult.hidden = false;
  document.querySelector("#empty-title").textContent = "Analyzing transaction";
  document.querySelector("#empty-copy").textContent =
    "The Random Forest is evaluating the transaction's features.";
  document.querySelector("#result-state").textContent = "In progress";
  formStatus.textContent = "Running the trained classifier";

  try {
    const result = await getJSON("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(transaction),
      signal: controller.signal,
    });
    if (version !== requestVersion) return;
    renderPrediction(result);
    formStatus.textContent = "Prediction complete";
  } catch (error) {
    if (version !== requestVersion) return;
    document.querySelector("#empty-title").textContent = "Analysis unavailable";
    document.querySelector("#empty-copy").textContent =
      "Check the message beside the form and try again.";
    document.querySelector("#result-state").textContent = "Unable to analyze";
    const message =
      error.name === "AbortError"
        ? "The analysis took too long. Please try again."
        : error.message;
    showError(message, error.fields);
    formStatus.textContent = "Analysis could not be completed";
  } finally {
    clearTimeout(timeout);
    if (version === requestVersion) {
      activePrediction = null;
      setBusy(false);
    }
  }
});

async function loadModel() {
  const status = document.querySelector("#model-status");
  try {
    const model = await getJSON("/api/model");
    if (!model.ready)
      throw new Error(model.error || "The saved model is unavailable.");
    modelReady = true;
    document.querySelector("#model-name").textContent = model.name;
    document.querySelector("#model-trees").textContent = model.trees;
    document.querySelector("#model-features").textContent = model.features;
    document.querySelector("#model-threshold").textContent =
      `${model.threshold * 100}%`;
    document.querySelector("#model-status-text").textContent = "Model ready";
    status.classList.add("ready");
    setBusy(false);
  } catch (error) {
    document.querySelector("#model-status-text").textContent =
      "Model unavailable";
    status.classList.add("error");
    const modelError = document.querySelector("#model-error");
    modelError.textContent = error.message;
    modelError.hidden = false;
    formStatus.textContent = "Restore the model and reload this page";
  }
}

async function loadExamples() {
  const message = document.querySelector("#example-message");
  try {
    const data = await getJSON("/api/examples");
    if (!data.examples.length)
      throw new Error(
        "Example transactions are unavailable. Enter your transaction values below.",
      );
    exampleButtons.replaceChildren();
    data.examples.forEach((example) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "example-button";
      button.textContent = example.name;
      button.addEventListener("click", () => {
        invalidatePrediction();
        fields.forEach((field) => {
          field.value = example.values[field.name] ?? "";
        });
        updateCompleteness();
        message.textContent = `${example.name} loaded. Select Analyze transaction to run the model.`;
        message.hidden = false;
      });
      exampleButtons.append(button);
    });
  } catch (error) {
    exampleButtons.replaceChildren();
    message.textContent = error.message;
    message.hidden = false;
  }
}

loadModel();
loadExamples();

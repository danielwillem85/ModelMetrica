const tabs = document.querySelectorAll(".tab");
const panels = document.querySelectorAll(".tab-panel");
const processingOverlay = document.getElementById("processing-overlay");
const processingProgress = document.getElementById("processing-progress");
const processingProgressFill = document.getElementById("processing-progress-fill");
const processingProgressText = document.getElementById("processing-progress-text");
const processingModelListWrap = document.getElementById("processing-model-list-wrap");
const processingModelList = document.getElementById("processing-model-list");
const modelSelectionOverlay = document.getElementById("model-selection-overlay");
const subscriptionOverlay = document.getElementById("subscription-overlay");
const accountSubscriptionOverlay = document.getElementById("account-subscription-overlay");
const hasSubscription = document.body.dataset.hasSubscription === "true";
const signupToggle = document.querySelector("[data-show-signup]");
const signupForm = document.querySelector("[data-signup-form]");
const signupPassword = document.getElementById("signup_password");
const signupConfirmPassword = document.getElementById("signup_confirm_password");
let processingProgressTimer = null;

function showProcessingOverlay(totalModels = 1) {
  if (processingOverlay) {
    resetProcessingProgress(totalModels);
    processingOverlay.classList.add("active");
  }
}

function stopProcessingProgressPolling() {
  if (processingProgressTimer) {
    window.clearInterval(processingProgressTimer);
    processingProgressTimer = null;
  }
}

function resetProcessingProgress(totalModels) {
  updateProcessingProgress(0, 0, totalModels, []);
}

function updateProcessingProgress(percent, completed, total, labels = []) {
  const safeTotal = Math.max(Number(total) || 0, 0);
  const safeCompleted = Math.max(Number(completed) || 0, 0);
  const safePercent = Math.max(0, Math.min(100, Math.round(Number(percent) || 0)));
  const fittedLabels = Array.isArray(labels) ? labels.filter(Boolean) : [];
  if (processingProgressFill) {
    processingProgressFill.style.width = `${safePercent}%`;
  }
  if (processingProgress) {
    processingProgress.setAttribute("aria-valuenow", String(safePercent));
  }
  if (processingProgressText) {
    const totalText = safeTotal || 1;
    const completedText = Math.min(safeCompleted, totalText);
    processingProgressText.textContent = `${completedText} of ${totalText} model(s) estimated.`;
  }
  if (processingModelList && processingModelListWrap) {
    processingModelList.replaceChildren();
    fittedLabels.forEach((label) => {
      const item = document.createElement("li");
      item.textContent = label;
      processingModelList.appendChild(item);
    });
    processingModelListWrap.hidden = fittedLabels.length === 0;
  }
}

function startProcessingProgressPolling(jobId, fallbackTotal) {
  if (!jobId) {
    return;
  }
  stopProcessingProgressPolling();
  const poll = () => {
    fetch(`/run-progress/${encodeURIComponent(jobId)}`, { headers: { Accept: "application/json" } })
      .then((response) => (response.ok ? response.json() : null))
      .then((progress) => {
        if (!progress) {
          return;
        }
        const total = progress.total || fallbackTotal;
        const completed = progress.completed || 0;
        const percent = progress.percent || (total ? (completed / total) * 100 : 0);
        updateProcessingProgress(percent, completed, total, progress.labels || []);
        if (progress.status === "complete") {
          stopProcessingProgressPolling();
        }
      })
      .catch(() => {});
  };
  poll();
  processingProgressTimer = window.setInterval(poll, 500);
}

function selectedModelCount(form) {
  const modelCheckboxes = form.querySelectorAll(".model-choice-grid input[type='checkbox']");
  if (modelCheckboxes.length) {
    return Math.max(Array.from(modelCheckboxes).filter((checkbox) => checkbox.checked).length, 1);
  }
  const modelSelect = form.querySelector("select[name$='_model']");
  if (!modelSelect) {
    return 1;
  }
  return Math.max(Array.from(modelSelect.selectedOptions).filter((option) => option.value).length, 1);
}

function runProgressId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function ensureRunProgressInput(form) {
  let input = form.querySelector("input[name='run_progress_id']");
  if (!input) {
    input = document.createElement("input");
    input.type = "hidden";
    input.name = "run_progress_id";
    form.appendChild(input);
  }
  input.value = runProgressId();
  return input.value;
}

function showModelSelectionOverlay() {
  if (modelSelectionOverlay) {
    modelSelectionOverlay.classList.add("active");
  }
}

function hideModelSelectionOverlay() {
  if (modelSelectionOverlay) {
    modelSelectionOverlay.classList.remove("active");
  }
}

function showSubscriptionOverlay() {
  if (subscriptionOverlay) {
    subscriptionOverlay.classList.add("active");
  }
}

function hideSubscriptionOverlay() {
  if (subscriptionOverlay) {
    subscriptionOverlay.classList.remove("active");
  }
}

function showAccountSubscriptionOverlay() {
  if (accountSubscriptionOverlay) {
    accountSubscriptionOverlay.classList.add("active");
  }
}

function hideAccountSubscriptionOverlay() {
  if (accountSubscriptionOverlay) {
    accountSubscriptionOverlay.classList.remove("active");
  }
}

function activateTab(hash) {
  const validTabs = ["#classification", "#regression", "#pro_classification", "#pro_regression"];
  const target = validTabs.includes(hash) ? hash.slice(1) : "data";
  tabs.forEach((tab) => tab.classList.toggle("active", tab.getAttribute("href") === `#${target}`));
  panels.forEach((panel) => panel.classList.toggle("active", panel.id === target));
}

tabs.forEach((tab) => {
  tab.addEventListener("click", (event) => {
    event.preventDefault();
    history.replaceState(null, "", tab.getAttribute("href"));
    activateTab(tab.getAttribute("href"));
  });
});

function validateSignupPasswords() {
  if (!signupPassword || !signupConfirmPassword) {
    return;
  }
  const mismatch = signupConfirmPassword.value && signupPassword.value !== signupConfirmPassword.value;
  signupConfirmPassword.setCustomValidity(mismatch ? "Passwords do not match." : "");
}

if (signupToggle && signupForm) {
  signupToggle.addEventListener("click", () => {
    signupForm.hidden = false;
    signupToggle.hidden = true;
    signupForm.querySelector("input[name='username']")?.focus();
  });
}

signupPassword?.addEventListener("input", validateSignupPasswords);
signupConfirmPassword?.addEventListener("input", validateSignupPasswords);

document.querySelectorAll(".model-comparison tbody tr[data-model-name]").forEach((row) => {
  row.addEventListener("click", () => {
    const panel = row.closest(".tab-panel");
    if (!panel) {
      return;
    }
    const detailSelect = panel.querySelector("select[name$='_detail_model']");
    const form = panel.querySelector("form");
    if (!detailSelect || !form) {
      return;
    }
    detailSelect.value = row.dataset.modelName;
    form.requestSubmit ? form.requestSubmit() : form.submit();
  });
});

document.querySelectorAll("form[data-run-form]").forEach((form) => {
  form.addEventListener("submit", (event) => {
    if (!formHasSelectedModel(form)) {
      event.preventDefault();
      showModelSelectionOverlay();
      return;
    }
    if (!form.checkValidity()) {
      return;
    }
    if (form.hasAttribute("data-pro-run-form") && !hasSubscription) {
      event.preventDefault();
      showSubscriptionOverlay();
      return;
    }
    const modelCount = selectedModelCount(form);
    const progressJobId = ensureRunProgressInput(form);
    showProcessingOverlay(modelCount);
    startProcessingProgressPolling(progressJobId, modelCount);
  });
});

function formHasSelectedModel(form) {
  const modelCheckboxes = form.querySelectorAll(".model-choice-grid input[type='checkbox']");
  if (modelCheckboxes.length) {
    return Array.from(modelCheckboxes).some((checkbox) => checkbox.checked);
  }
  const modelSelect = form.querySelector("select[name$='_model']");
  if (!modelSelect) {
    return true;
  }
  return Array.from(modelSelect.selectedOptions).some((option) => option.value);
}

document.querySelectorAll("[data-close-model-selection]").forEach((button) => {
  button.addEventListener("click", hideModelSelectionOverlay);
});

document.querySelectorAll("[data-close-subscription]").forEach((button) => {
  button.addEventListener("click", hideSubscriptionOverlay);
});

document.querySelectorAll("[data-open-account-subscription]").forEach((button) => {
  button.addEventListener("click", showAccountSubscriptionOverlay);
});

document.querySelectorAll("[data-close-account-subscription]").forEach((button) => {
  button.addEventListener("click", hideAccountSubscriptionOverlay);
});

if (subscriptionOverlay) {
  subscriptionOverlay.addEventListener("click", (event) => {
    if (event.target === subscriptionOverlay) {
      hideSubscriptionOverlay();
    }
  });
}

if (modelSelectionOverlay) {
  modelSelectionOverlay.addEventListener("click", (event) => {
    if (event.target === modelSelectionOverlay) {
      hideModelSelectionOverlay();
    }
  });
}

if (accountSubscriptionOverlay) {
  accountSubscriptionOverlay.addEventListener("click", (event) => {
    if (event.target === accountSubscriptionOverlay) {
      hideAccountSubscriptionOverlay();
    }
  });
}

document.querySelectorAll("[data-threshold-input]").forEach((input) => {
  const output = input.closest(".threshold-control")?.querySelector("[data-threshold-output]");
  const update = () => {
    if (output) {
      output.value = Number(input.value).toFixed(3);
      output.textContent = Number(input.value).toFixed(3);
    }
  };
  input.addEventListener("input", update);
  update();
});

if (window.location.hash) {
  activateTab(window.location.hash);
}

window.addEventListener("pageshow", () => {
  if (processingOverlay) {
    processingOverlay.classList.remove("active");
  }
  stopProcessingProgressPolling();
});

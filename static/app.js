const tabs = document.querySelectorAll(".tab");
const panels = document.querySelectorAll(".tab-panel");
const processingOverlay = document.getElementById("processing-overlay");
const subscriptionOverlay = document.getElementById("subscription-overlay");
const hasSubscription = document.body.dataset.hasSubscription === "true";
const signupToggle = document.querySelector("[data-show-signup]");
const signupForm = document.querySelector("[data-signup-form]");
const signupPassword = document.getElementById("signup_password");
const signupConfirmPassword = document.getElementById("signup_confirm_password");

function showProcessingOverlay() {
  if (processingOverlay) {
    processingOverlay.classList.add("active");
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
    if (!form.checkValidity()) {
      return;
    }
    if (form.hasAttribute("data-pro-run-form") && !hasSubscription) {
      event.preventDefault();
      showSubscriptionOverlay();
      return;
    }
    showProcessingOverlay();
  });
});

document.querySelectorAll("[data-close-subscription]").forEach((button) => {
  button.addEventListener("click", hideSubscriptionOverlay);
});

if (subscriptionOverlay) {
  subscriptionOverlay.addEventListener("click", (event) => {
    if (event.target === subscriptionOverlay) {
      hideSubscriptionOverlay();
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
});

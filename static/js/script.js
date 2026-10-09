/* =========================================================
   PawCare JavaScript - UI/UX & Interactions
   ========================================================= */

// 1. Password Show / Hide
function togglePassword(inputId, iconId) {
    const passwordInput = document.getElementById(inputId);
    const icon = document.getElementById(iconId);

    if (!passwordInput || !icon) {
        return;
    }

    if (passwordInput.type === "password") {
        passwordInput.type = "text";
        icon.textContent = "🙈";
    } else {
        passwordInput.type = "password";
        icon.textContent = "👁";
    }
}

// 2. Form Submit Loading States & UI Enhancements
document.addEventListener("DOMContentLoaded", function () {
    // Auto-dismiss standard alerts after 5 seconds
    const autoDismissAlerts = document.querySelectorAll(".alert-dismissible:not(.alert-persistent)");
    autoDismissAlerts.forEach(function (alertElem) {
        setTimeout(function () {
            if (alertElem && alertElem.parentNode) {
                if (typeof bootstrap !== "undefined" && bootstrap.Alert) {
                    try {
                        const bsAlert = bootstrap.Alert.getOrCreateInstance(alertElem);
                        if (bsAlert) {
                            bsAlert.close();
                            return;
                        }
                    } catch (err) {
                        // fallback to class removal
                    }
                }
                alertElem.classList.remove("show");
                setTimeout(function () {
                    if (alertElem.parentNode) {
                        alertElem.parentNode.removeChild(alertElem);
                    }
                }, 150);
            }
        }, 5000);
    });

    // Loading states on file upload & data submission forms
    const formsWithLoading = document.querySelectorAll("form.needs-loading, form[enctype='multipart/form-data'], form.action-loading");
    formsWithLoading.forEach(function (form) {
        form.addEventListener("submit", function (e) {
            // Check HTML5 validity
            if (form.checkValidity && !form.checkValidity()) {
                return;
            }

            const submitBtn = form.querySelector("button[type='submit'], input[type='submit']");
            if (submitBtn && !submitBtn.disabled) {
                const loadingText = submitBtn.getAttribute("data-loading-text") || "Processing...";
                
                // Allow a tiny delay so the submit event actually fires
                setTimeout(function () {
                    submitBtn.setAttribute("disabled", "true");
                    submitBtn.classList.add("disabled");
                    if (submitBtn.tagName.toLowerCase() === "button") {
                        submitBtn.innerHTML = `<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>${loadingText}`;
                    }
                }, 10);
            }
        });
    });

    // Photo input live preview helper
    const photoInputs = document.querySelectorAll("input[type='file'][data-preview-target]");
    photoInputs.forEach(function (input) {
        input.addEventListener("change", function () {
            const targetSelector = input.getAttribute("data-preview-target");
            const previewEl = document.querySelector(targetSelector);
            if (previewEl && input.files && input.files[0]) {
                const reader = new FileReader();
                reader.onload = function (e) {
                    if (previewEl.tagName.toLowerCase() === "img") {
                        previewEl.src = e.target.result;
                    } else {
                        previewEl.innerHTML = `<img src="${e.target.result}" style="width: 100%; height: 100%; object-fit: cover; border-radius: inherit;" alt="Preview">`;
                    }
                };
                reader.readAsDataURL(input.files[0]);
            }
        });
    });
});
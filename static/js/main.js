/**
 * IntelliSentiment AI — Main JavaScript
 */

"use strict";

// ---- Active nav link highlighting ----
(function highlightNav() {
  const path = window.location.pathname;
  document.querySelectorAll(".navbar .nav-link").forEach(link => {
    const href = link.getAttribute("href");
    if (href === path || (href !== "/" && path.startsWith(href))) {
      link.classList.add("active");
    }
  });
})();

// ---- Auto-dismiss alerts after 5 seconds ----
document.querySelectorAll(".alert.alert-dismissible").forEach(alert => {
  setTimeout(() => {
    const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
    if (bsAlert) bsAlert.close();
  }, 5000);
});

// ---- Tooltip initialisation ----
document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => {
  new bootstrap.Tooltip(el);
});


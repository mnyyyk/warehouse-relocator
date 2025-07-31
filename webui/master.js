/**
 * master.js
 * ----------
 * Handles the upload of master files (SKU, receipts, shipments) from
 * `master_upload.html`. Sends a multipart/form‐data POST request to
 * the `/uploads/` API endpoint and reports success / failure to the user.
 */

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("master-upload-form");
  const statusEl = document.getElementById("upload-status");

  if (!form) {
    // Defensive guard – the script might be loaded on another page.
    return;
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    statusEl.textContent = "Uploading…";
    statusEl.className = "";

    // Map each input element to the name FastAPI expects
    const inputsMap = [
      { el: document.getElementById("sku-file"), fieldName: "sku_file" },
      { el: document.getElementById("receipts-file"), fieldName: "receipts_file" },
      { el: document.getElementById("shipments-file"), fieldName: "shipments_file" },
    ];

    // Find which input actually has a file selected
    let selected = inputsMap.find(
      ({ el }) => el && el.files && el.files.length > 0
    );

    if (!selected) {
      statusEl.textContent = "Please choose a file.";
      statusEl.className = "error";
      return;
    }

    const formData = new FormData();
    formData.append(selected.fieldName, selected.el.files[0]);

    try {
      const response = await fetch("/uploads/master/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(errText || `Upload failed with status ${response.status}`);
      }

      statusEl.textContent = "✅ Upload successful!";
      statusEl.className = "success";
      form.reset();
    } catch (err) {
      console.error(err);
      statusEl.textContent = `❌ ${err.message}`;
      statusEl.className = "error";
    }
  });
});

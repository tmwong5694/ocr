const form = document.querySelector<HTMLFormElement>("#upload-form")!;
const apiUrlInput = document.querySelector<HTMLInputElement>("#api-url")!;
const fileInput = document.querySelector<HTMLInputElement>("#file-input")!;
const statusEl = document.querySelector<HTMLParagraphElement>("#status")!;
const outputEl = document.querySelector<HTMLPreElement>("#output")!;
const submitBtn = document.querySelector<HTMLButtonElement>("#submit-btn")!;

const setStatus = (message: string) => {
  statusEl.textContent = message;
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = fileInput.files?.[0];
  if (!file) {
    setStatus("Choose a PDF first.");
    return;
  }

  const apiUrl = apiUrlInput.value.trim();
  const formData = new FormData();
  formData.append("file", file);

  submitBtn.disabled = true;
  setStatus("Uploading and running OCR...");
  outputEl.textContent = "";

    const startedAt = performance.now();

    try {
    const response = await fetch(apiUrl, {
      method: "POST",
      body: formData,
    });

    const payload = await response.json().catch(() => null);

    if (!response.ok) {
      const detail =
        payload && typeof payload === "object" && "detail" in payload
          ? String((payload as { detail?: unknown }).detail ?? "")
          : "";

      throw new Error(
        detail || `Request failed with status ${response.status} ${response.statusText}`,
      );
    }

    const seconds = ((performance.now() - startedAt) / 1000).toFixed(2);
    setStatus(`It took ${seconds} seconds to complete extraction`);
    outputEl.textContent = JSON.stringify(payload, null, 2);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    setStatus(`Error: ${message}`);
    outputEl.textContent = message;
  } finally {
    submitBtn.disabled = false;
  }
});
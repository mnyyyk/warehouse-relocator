/* webui/analysis.js
   ---------------------------------------------------------
   Front‑end helper that lets the user trigger an analysis run
   and then shows the results in a simple table.

   Expected HTML elements
   ----------------------
   * A button  with  id="run‑analysis‑btn"
   * A  <div>  with  id="analysis‑spinner" (optional, shown while running)
   * A  <pre>  or <div> with  id="analysis‑output" where we will inject a
     JSON‑pretty‑printed summary for now.

   Feel free to improve the UI later; this keeps things minimal while the
   back‑end endpoints stabilise.
*/
(() => {
  // -------------------------------------------------------------------
  // DOM helpers
  // -------------------------------------------------------------------
  const $ = (sel) => document.querySelector(sel);
  const btnRun     = $('#run-analysis-btn');
  const boxSpinner = $('#analysis-spinner');
  const boxOutput  = $('#analysis-output');

  const showSpinner = (show = true) => {
    if (!boxSpinner) return;
    boxSpinner.style.display = show ? 'block' : 'none';
  };

  const prettify = (obj) =>
    JSON.stringify(obj, null, 2)
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

  /**
   * Turn a flat object (key‑value pairs) into a two‑column HTML table.
   * Nested objects/arrays are JSON‑stringified for brevity.
   */
  const tableFromObj = (obj) => {
    const rows = Object.entries(obj)
      .map(
        ([k, v]) =>
          `<tr><th>${k}</th><td>${
            typeof v === 'object' ? prettify(v) : String(v)
          }</td></tr>`
      )
      .join('');
    return `<table class="analysis-table">${rows}</table>`;
  };

  // -------------------------------------------------------------------
  // REST helpers
  // -------------------------------------------------------------------
  /**
   * Fetch the most recent analysis result, if any.
   */
  async function fetchLatestResult() {
    const res = await fetch('/analysis/latest', { headers: { Accept: 'application/json' } });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }

  /**
   * Trigger a new analysis run.
   */
  async function runAnalysis() {
    const res = await fetch('/analysis/run', {
      method: 'POST',
      headers: { Accept: 'application/json' }
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }

  // -------------------------------------------------------------------
  // UI actions
  // -------------------------------------------------------------------
  async function handleRunClick() {
    try {
      btnRun.disabled = true;
      showSpinner(true);
      const result = await runAnalysis();
      boxOutput.innerHTML = tableFromObj(result);
    } catch (err) {
      console.error(err);
      alert(`分析の実行に失敗しました: ${err.message ?? err}`);
    } finally {
      showSpinner(false);
      btnRun.disabled = false;
    }
  }

  async function initialise() {
    // Wire‑up handler
    if (btnRun) btnRun.addEventListener('click', handleRunClick);

    // Try to show the latest cached result on page load
    try {
      const latest = await fetchLatestResult();
      if (latest && Object.keys(latest).length) {
        boxOutput.innerHTML = tableFromObj(latest);
      }
    } catch {
      /* no cached result yet – ignore */
    }
  }

  // kick‑off once DOM is ready
  if (document.readyState !== 'loading') {
    initialise();
  } else {
    document.addEventListener('DOMContentLoaded', initialise);
  }
})();

// Read the chart values from the published Table 2 so the chart and table agree.
function renderBars(metric) {
  const chart = document.getElementById('bars');
  const results = [...document.querySelectorAll('#transfer-table tbody tr')];
  chart.replaceChildren(...results.map(result => {
    const value = Number(result.querySelector(`[data-metric="${metric}"]`).textContent);
    const row = document.createElement('div');
    row.className = `bar-row ${result.className}`;
    const label = document.createElement('div');
    label.className = 'bar-label';
    const name = document.createElement('span');
    name.textContent = result.querySelector('th').textContent;
    const score = document.createElement('strong');
    score.textContent = value.toFixed(2);
    label.append(name, score);
    const track = document.createElement('div');
    track.className = 'bar-track';
    track.setAttribute('aria-hidden', 'true');
    const bar = document.createElement('div');
    bar.className = 'bar';
    bar.style.width = `${value}%`;
    track.append(bar);
    row.append(label, track);
    return row;
  }));
  document.getElementById('chart-domain').textContent = `${metric === 'camus' ? 'Source' : 'Target'} domain · Dice (%) · Higher is better`;
}
document.querySelectorAll('button[data-metric]').forEach(button => {
  button.addEventListener('click', () => {
    document.querySelectorAll('button[data-metric]').forEach(item => {
      const selected = item === button;
      item.classList.toggle('selected', selected);
      item.setAttribute('aria-pressed', String(selected));
    });
    renderBars(button.dataset.metric);
  });
});

// All 380 Dice/HD95 values are transcribed from arXiv v2, Table 1.
const supervised = JSON.parse(document.getElementById('supervised-data').textContent);
function renderSupervised() {
  const dataset = supervised.datasets.find(item => item.name === document.getElementById('supervised-dataset').value);
  const budget = document.getElementById('supervised-budget').value;
  const rows = supervised.rows.filter(row => row.budget === budget);
  const table = document.getElementById('supervised-table');
  const caption = document.createElement('caption');
  caption.textContent = `Table 1 · ${dataset.name} · ${budget} supervision. Dice (%) ↑ and HD95 (mm) ↓, reported per structure.`;
  const head = document.createElement('thead');
  const headings = document.createElement('tr');
  const metrics = document.createElement('tr');
  const method = document.createElement('th');
  method.textContent = 'Method';
  method.scope = 'col';
  method.rowSpan = 2;
  headings.append(method);
  dataset.classes.forEach(name => {
    const cell = document.createElement('th');
    cell.textContent = name;
    cell.scope = 'colgroup';
    cell.colSpan = 2;
    headings.append(cell);
    ['Dice ↑', 'HD95 ↓'].forEach(label => {
      const metric = document.createElement('th');
      metric.textContent = label;
      metric.scope = 'col';
      metrics.append(metric);
    });
  });
  head.append(headings, metrics);
  const body = document.createElement('tbody');
  rows.forEach(result => {
    const row = document.createElement('tr');
    if (result.method.startsWith('Ours')) row.className = 'ours';
    const name = document.createElement('th');
    name.scope = 'row';
    name.textContent = result.method;
    row.append(name);
    result.values.slice(dataset.offset, dataset.offset + dataset.classes.length * 2).forEach(value => {
      const cell = document.createElement('td');
      cell.textContent = value.toFixed(2);
      row.append(cell);
    });
    body.append(row);
  });
  table.replaceChildren(caption, head, body);
  document.getElementById('supervised-status').textContent = `Showing ${dataset.name}, ${budget} supervision, ${rows.length} methods.`;
}
document.querySelectorAll('#supervised-dataset, #supervised-budget').forEach(control => {
  control.addEventListener('change', renderSupervised);
});

document.querySelectorAll('[data-copy]').forEach(button => {
  button.addEventListener('click', async () => {
    const target = document.getElementById(button.dataset.copy);
    const original = button.textContent;
    try {
      await navigator.clipboard.writeText(target.textContent);
      button.textContent = 'Copied!';
      document.getElementById('copy-status').textContent = 'Copied to clipboard.';
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(target);
      selection.removeAllRanges();
      selection.addRange(range);
      button.textContent = 'Text selected';
      document.getElementById('copy-status').textContent = 'Clipboard unavailable. Text selected; use your keyboard to copy.';
    }
    setTimeout(() => { button.textContent = original; }, 2500);
  });
});
renderBars('cardiacuda');
renderSupervised();

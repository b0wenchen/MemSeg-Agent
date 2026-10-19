const results = [
  { name: 'nnU-Net', avg: 22.65, lv: 14.65, la: 30.64 },
  { name: 'SwinUNETR', avg: 16.82, lv: 33.57, la: 0.07 },
  { name: 'U-Net', avg: 10.15, lv: 20.30, la: 0.03 },
  { name: 'MedSAM2', avg: 42.75, lv: 47.26, la: 38.31 },
  { name: 'Ours · Static', avg: 30.74, lv: 39.75, la: 21.72, className: 'static-row' },
  { name: 'Ours · Static + WM', avg: 77.30, lv: 81.53, la: 73.07, className: 'ours-row' },
];
function renderBars(metric) {
  const chart = document.getElementById('bars');
  chart.replaceChildren(...results.map(result => {
    const row = document.createElement('div');
    row.className = `bar-row ${result.className || ''}`;
    const label = document.createElement('div');
    label.className = 'bar-label';
    const name = document.createElement('span');
    name.textContent = result.name;
    const score = document.createElement('strong');
    score.textContent = result[metric].toFixed(2);
    label.append(name, score);
    const track = document.createElement('div');
    track.className = 'bar-track';
    track.setAttribute('aria-hidden', 'true');
    const bar = document.createElement('div');
    bar.className = 'bar';
    bar.style.width = `${result[metric]}%`;
    track.append(bar);
    row.append(label, track);
    return row;
  }));
}
document.querySelectorAll('[data-metric]').forEach(button => {
  button.addEventListener('click', () => {
    document.querySelectorAll('[data-metric]').forEach(item => {
      const selected = item === button;
      item.classList.toggle('selected', selected);
      item.setAttribute('aria-pressed', String(selected));
    });
    renderBars(button.dataset.metric);
  });
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
renderBars('avg');

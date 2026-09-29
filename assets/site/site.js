// DrugDis project page: framework panels, mobile menu, copy button.
// Every number shown here is taken from the manuscript and its canonical tables
// (results/tables): M0 under held-out cell lines from T07, cross-assay
// reproducibility from T03.

const mini = (kind) => `<div class="mx-cells ${kind}" aria-hidden="true">${
  Array.from({length: 24}, (_, i) => `<i class="c${i % 6} r${Math.floor(i / 6)}"></i>`).join('')}</div>`;

const bar = (label, value, cls, text) => `
  <div class="dd-bar-row"><span>${label}</span>
    <div class="dd-bar-track"><i class="${cls}" style="--w:${value * 100}%"></i></div>
    <b>${text}</b></div>`;

const visuals = {
  decomposition: () => `
    <div class="diag-vis">
      <div class="diag-vis-heading"><span>One operator, built on the observed pairs</span><span>y = Hy + My</span></div>
      <div class="mx-row">
        <div class="mx"><span>total response</span>${mini('total')}</div>
        <b class="mx-op">=</b>
        <div class="mx"><span>additive component</span>${mini('add')}</div>
        <b class="mx-op">+</b>
        <div class="mx"><span>interaction component</span>${mini('int')}</div>
      </div>
      <div class="criterion-band">
        <span><b>Hy</b> = best additive fit of drug and sample marginals</span>
        <span><b>My</b> = y − Hy, orthogonal to every drug and every sample</span>
        <strong>Var(y) = Var(Hy) + Var(My), exactly</strong>
      </div>
      <div class="vis-caption">On the benchmark dataset the additive subspace occupies 1.76% of the degrees of freedom and carries 75.8% of response variance.</div>
    </div>`,
  recovery: () => `
    <div class="diag-vis">
      <div class="diag-vis-heading"><span>M0, held-out cell lines, mean of five seeds</span><span>correlation</span></div>
      <div class="dd-bars">
        ${bar('total response', 0.86, 'tot', '0.86')}
        ${bar('additive component', 0.96, 'add', '0.96')}
        ${bar('interaction component', 0.32, 'int', '0.32')}
      </div>
      <div class="criterion-band compact">
        <span>measured and predicted decomposed by the same operator</span>
        <span>correlation within each component</span>
        <strong>aggregate 0.86 is not interaction 0.32</strong>
      </div>
      <div class="vis-caption">Correlations on the same test predictions; the five test supports hold 474,279–570,863 pairs.</div>
    </div>`,
  error: () => `
    <div class="diag-vis">
      <div class="diag-vis-heading"><span>M0, held-out cell lines, mean of five seeds</span><span>squared prediction error</span></div>
      <div class="dd-stack" aria-hidden="true">
        <i class="add" style="--w:23.3%"><span>0.0014</span></i><i class="int" style="--w:76.7%"><span>0.0046</span></i>
      </div>
      <div class="dd-stack-legend"><span><i class="add"></i>additive-component error</span><span><i class="int"></i>interaction-component error</span><span>total 0.0060</span></div>
      <div class="criterion-band">
        <span><b>A<sub>int</sub></b> = sd(M ŷ) / sd(My) = 0.35</span>
        <span><b>MSE</b> = E<sub>add</sub> + E<sub>int</sub>, exactly</span>
        <strong>R²<sub>int</sub> = 2·A<sub>int</sub>·r<sub>int</sub> − A<sub>int</sub>²</strong>
      </div>
      <div class="vis-caption">Under held-out cell lines, 77% of M0's squared error lies in the interaction component, and the predicted interaction has about a third of the measured spread.</div>
    </div>`,
  reference: () => `
    <div class="diag-vis">
      <div class="diag-vis-heading"><span>32,724 pairs measured in both GDSC1 and GDSC2</span><span>cross-assay reproducibility</span></div>
      <div class="dd-bars">
        ${bar('additive component', 0.75, 'add', '0.75 (√ 0.87)')}
        ${bar('interaction component', 0.50, 'int', '0.50 (√ 0.71)')}
      </div>
      <div class="criterion-band compact">
        <span><b>q<sub>c</sub></b> = agreement of component c between assays</span>
        <span><b>√q<sub>c</sub></b> = empirical recovery reference</span>
        <strong>a reference for matched supports, not a ceiling</strong>
      </div>
      <div class="vis-caption">On matched supports under held-out cell lines, M0's additive recovery was 1.00 times its reference and its interaction recovery 0.37 times its own.</div>
    </div>`
};

const panels = {
  decomposition: {
    number: '01 / Orthogonal decomposition',
    title: 'Which part of the response is additive, and which is interaction?',
    copy: 'On the observed drug–sample pairs, the total response splits exactly into an additive component, the best additive fit of drug and sample marginals, and a drug–sample interaction component orthogonal to it.',
    note: 'The interaction component is defined relative to the observed pairs: measuring different compounds, samples or pairs changes it. It is not an intrinsic biological interaction.',
    visual: visuals.decomposition
  },
  recovery: {
    number: '02 / Component recovery',
    title: 'Does a prediction recover each component, or only their sum?',
    copy: 'Measured and predicted responses are decomposed by the same operator. Correlation within each component reports whether the prediction points in the right direction for additive effects and for interactions separately.',
    note: 'A high total-response correlation can be carried almost entirely by the additive component.',
    visual: visuals.recovery
  },
  error: {
    number: '03 / Amplitude and error',
    title: 'How large is the predicted interaction, and where does the error lie?',
    copy: 'Amplitude compares predicted and measured interaction spread; squared prediction error splits exactly into an additive and an interaction part; interaction R² combines direction and amplitude.',
    note: 'Direction alone is not enough: in zero-shot organoids, interaction R² stayed below zero for every arm.',
    visual: visuals.error
  },
  reference: {
    number: '04 / Reproducibility reference',
    title: 'How much of each component do repeated measurements reproduce?',
    copy: 'Where the same pairs were measured twice, as in GDSC1 and GDSC2, the agreement of each component between assays gives an empirical reference for interpreting recovery.',
    note: 'The reference characterizes matched supports. It is not an attainable limit on model performance.',
    visual: visuals.reference
  }
};

const style = document.createElement('style');
style.textContent = `
.evidence-graphic{height:auto;min-height:250px;margin:26px 0 18px;border:0!important;background:none!important}
.evidence-graphic:before,.evidence-graphic:after{display:none!important}
.evidence-graphic>.axis-x,.evidence-graphic>.axis-y{display:none!important}
.diag-vis{border-top:1px solid #1e1e1e;border-bottom:1px solid #d8d4cc;background:#fff;padding:13px 0 0;color:#343434}
.diag-vis-heading{display:flex;justify-content:space-between;gap:20px;padding:0 3px 10px;font-size:8px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#6f6d69}
.criterion-band{display:grid;grid-template-columns:1fr 1.35fr 1.2fr;border-top:1px solid #d8d4cc;background:#f6f4ef}.criterion-band>*{padding:11px 12px;border-right:1px solid #d8d4cc;font-size:8px;line-height:1.45}.criterion-band>*:last-child{border-right:0}.criterion-band strong{color:#8f2e28}.criterion-band.compact{grid-template-columns:1fr 1fr 1.65fr}
.vis-caption{padding:10px 3px 12px;font-size:9px;line-height:1.55;color:#6f6d69}
.mx-row{display:grid;grid-template-columns:1fr 22px 1fr 22px 1fr;align-items:center;gap:6px;padding:22px 10px 20px;border-top:1px solid #e2ded6}
.mx{text-align:center}.mx>span{display:block;margin-bottom:8px;font-size:8px;color:#6f6d69;text-transform:uppercase;letter-spacing:.08em}
.mx-op{font:400 22px/1 Georgia,serif;color:#b43a32;text-align:center}
.mx-cells{display:grid;grid-template-columns:repeat(6,1fr);gap:2px;width:112px;margin:auto}
.mx-cells i{display:block;aspect-ratio:1;background:#e7e4de}
.mx-cells.total i{background:#8d8a84}.mx-cells.total i.c1,.mx-cells.total i.c4{background:#b9b6af}.mx-cells.total i.r1.c2,.mx-cells.total i.r3.c5{background:#5c5a56}.mx-cells.total i.r2.c0,.mx-cells.total i.r0.c3{background:#e7e4de}
.mx-cells.add i{background:var(--dd-add-2)}.mx-cells.add i.c1,.mx-cells.add i.c4{background:var(--dd-add-3)}.mx-cells.add i.c2{background:var(--dd-add)}.mx-cells.add i.r2{filter:brightness(1.12)}
.mx-cells.int i{background:#f1e6ef}.mx-cells.int i.r1.c2,.mx-cells.int i.r3.c5,.mx-cells.int i.r0.c0{background:var(--dd-int)}.mx-cells.int i.r2.c0,.mx-cells.int i.r0.c3,.mx-cells.int i.r3.c1{background:#d7b5d3}
.dd-bars{display:grid;gap:12px;padding:22px 14px 20px;border-top:1px solid #e2ded6}
.dd-bar-row{display:grid;grid-template-columns:130px 1fr 88px;gap:12px;align-items:center;font-size:9px}
.dd-bar-row b{font:400 15px/1 Georgia,serif;color:#1e1e1e;white-space:nowrap}
.dd-bar-track{position:relative;height:14px;background:#efede8}.dd-bar-track i{position:absolute;left:0;top:0;bottom:0;width:var(--w)}
.dd-bar-track i.tot,.dd-stack-legend i.tot{background:#6f6d69}.dd-bar-track i.add,.dd-stack i.add,.dd-stack-legend i.add{background:var(--dd-add)}.dd-bar-track i.int,.dd-stack i.int,.dd-stack-legend i.int{background:var(--dd-int)}
.dd-stack{display:flex;height:30px;margin:22px 14px 8px;border-top:0}.dd-stack i{display:flex;align-items:center;justify-content:center;width:var(--w)}.dd-stack i span{font-size:9px;font-weight:700;color:#fff}
.dd-stack-legend{display:flex;gap:18px;flex-wrap:wrap;padding:0 14px 16px;font-size:8px;color:#6f6d69}.dd-stack-legend i{display:inline-block;width:9px;height:9px;margin-right:5px;vertical-align:-1px}
@media(max-width:650px){.criterion-band,.criterion-band.compact{grid-template-columns:1fr}.criterion-band>*{border-right:0;border-bottom:1px solid #d8d4cc}.criterion-band>*:last-child{border-bottom:0}.mx-row{grid-template-columns:1fr 14px 1fr 14px 1fr;padding-left:2px;padding-right:2px}.mx-cells{width:72px}.dd-bar-row{grid-template-columns:96px 1fr 72px}.diag-vis-heading{font-size:7px}}
`;
document.head.appendChild(style);

function renderPanel(key) {
  const panel = panels[key];
  if (!panel) return;
  document.querySelector('#panel-number').textContent = panel.number;
  document.querySelector('#panel-title').textContent = panel.title;
  document.querySelector('#panel-copy').textContent = panel.copy;
  document.querySelector('#panel-note').textContent = panel.note;
  const visual = document.querySelector('.evidence-graphic');
  if (visual) visual.innerHTML = panel.visual();
}

document.querySelectorAll('.diagnostic').forEach(button => {
  button.addEventListener('click', () => {
    document.querySelectorAll('.diagnostic').forEach(item => {
      item.classList.toggle('active', item === button);
      item.setAttribute('aria-selected', String(item === button));
    });
    renderPanel(button.dataset.panel);
  });
});
renderPanel('decomposition');

const menuButton = document.querySelector('.menu-toggle');
const nav = document.querySelector('#site-nav');
menuButton?.addEventListener('click', () => {
  const open = nav.classList.toggle('is-open');
  menuButton.setAttribute('aria-expanded', String(open));
});
nav?.querySelectorAll('a').forEach(link => link.addEventListener('click', () => {
  nav.classList.remove('is-open');
  menuButton?.setAttribute('aria-expanded', 'false');
}));

document.querySelectorAll('.copy-button').forEach(button => button.addEventListener('click', async () => {
  const target = document.querySelector(`#${button.dataset.copyTarget}`);
  if (!target) return;
  try {
    await navigator.clipboard.writeText(target.innerText);
    button.textContent = 'Copied';
    setTimeout(() => (button.textContent = 'Copy'), 1600);
  } catch {
    button.textContent = 'Select code';
  }
}));

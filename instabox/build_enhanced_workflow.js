// Build "Mystic V5 - 文生视频 + 自动扩写" in the live ComfyUI and save it to the user's workflows:
// the Mystic FL2VA workflow plus H3 Guide (idea + duration) -> Prompt Enhancer (Heretic tail)
// -> video prompt, with a PreviewAny showing the expanded text.
//   node build_enhanced_workflow.js https://<public url> <user> <password>
const { chromium } = require('playwright');
(async () => {
  const [url, user, pass] = process.argv.slice(2);
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1600, height: 1000 } });
  await p.goto(url + '/login');
  await p.fill('input[name=user]', user); await p.fill('input[name=password]', pass); await p.click('button');
  await p.waitForFunction(() => window.app?.graph, null, { timeout: 60000 });
  const res = await p.evaluate(async () => {
    const src = 'Mystic V5 - 文生视频 (FL2VA).json';
    const dst = 'Mystic V5 - 文生视频 + 自动扩写.json';
    const wf = await (await fetch('/api/userdata/' + encodeURIComponent('workflows/' + src))).json();
    await window.app.loadGraphData(wf);
    const g = window.app.graph, LG = window.LiteGraph;
    const clip = g.nodes.find(n => n.type === 'CLIPLoader');
    const i2v = g.nodes.find(n => n.type === 'MiniMaxH3ImageToVideo');
    const add = (type, pos, title) => { const n = LG.createNode(type); n.pos = pos; if (title) n.title = title; g.add(n); return n; };
    const setw = (node, name, value) => { node.widgets.find(w => w.name === name).value = value; };
    const slot = (node, name) => node.inputs.findIndex(i => i.name === name);
    const out = (node, name) => node.outputs.findIndex(o => o.name === name);
    // One place for the duration: the Guide's h3_length replaces the Float + math nodes.
    for (const n of g.nodes.filter(n => n.title === 'Float (duration)' || n.type === 'ComfyMathExpression')) g.remove(n);
    const guide = add('MiniMaxH3PromptGuide', [180, 1120], '① 写你的想法和时长（自动扩写）');
    setw(guide, 'what_do_you_want', 'Create a video from text only');
    setw(guide, 'target_description', '一只橘猫在洒满阳光的窗台上伸懒腰，窗外传来鸟叫声，镜头缓慢推近。');
    setw(guide, 'duration_seconds', 10);
    const tail = add('MiniMaxH3GenerationTailLoader', [700, 1120], '扩写模型（Heretic 生成尾）');
    setw(tail, 'tail_file', 'qwen3vl_32b_h3_generation_tail_50_63_int8_convrot.safetensors');
    const enh = add('MiniMaxH3PromptEnhancer', [700, 1260], '扩写（去审查 Qwen3-VL-32B）');
    const prev = add('PreviewAny', [1230, 700], '② 扩写结果（运行后显示，可复制）');
    const links = [
      clip.connect(0, enh, slot(enh, 'clip')),
      tail.connect(0, enh, slot(enh, 'clip_tail')),
      guide.connect(out(guide, 'h3_prompt'), enh, slot(enh, 'manual_prompt')),
      guide.connect(out(guide, 'mode_report'), enh, slot(enh, 'mode_report')),
      guide.connect(out(guide, 'h3_length'), i2v, slot(i2v, 'length')),
      enh.connect(out(enh, 'enhanced_prompt'), i2v, slot(i2v, 'prompt')),
      enh.connect(out(enh, 'enhanced_prompt'), prev, 0),
    ];
    const graph = window.app.graph.serialize();
    const r = await fetch('/api/userdata/' + encodeURIComponent('workflows/' + dst) + '?overwrite=true&full_info=true',
      { method: 'POST', body: JSON.stringify(graph), headers: { 'Content-Type': 'application/json' } });
    return { links: links.map(l => !!l), save: r.status, nodes: graph.nodes.length, types: graph.nodes.map(n => n.type).sort().join(',') };
  });
  console.log(JSON.stringify(res));
  await b.close();
})();

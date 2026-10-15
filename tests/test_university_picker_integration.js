// Run with: node --test tests/test_university_picker*.js
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const html = readFileSync(resolve(__dirname, '../Hooke/webui/static/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function picker() {
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      value: '', textContent: '', innerHTML: '', disabled: false,
      style: {}, classList: { toggle() {} }, children: [], listeners: {},
      replaceChildren(...children) {
        this.children = children;
        this.value = children[0]?.value || '';
        this.innerHTML = '';
      },
      add(option) { this.children.push(option); },
      addEventListener(event, listener) { (this.listeners[event] ||= []).push(listener); },
    });
    return elements.get(id);
  };
  const display = (name, university, subject, facility) => ({
    name, category: 'real_lab', robot: 'ur5e', description: `${facility} (display only)`,
    navigation: {
      university, category: subject, category_label: subject, facility,
      task_label: facility, video: `/docs-assets/${name}.mp4`, url: null,
    },
  });
  const fixture = {
    robots: { ur5e: { category: 'robot_arm' } },
    tasks: [
      { name: 'real_lab_original', category: 'real_labs', robot: 'lab_instruments',
        navigation: { url: '/real-labs?scene=original' } },
      display('real_lab_shared_display', 'Alpha University', 'Chemistry', 'Optics Room'),
      display('real_lab_zeta_display', 'Zeta University', 'Biology', 'Imaging Core'),
    ],
    university: {
      ranking: { edition: 2027 },
      universities: [{ id: 'alpha', name: 'Alpha University' }, { id: 'empty', name: 'Empty University' }],
      labs: [{ scene_id: 'original', task_name: 'real_lab_original', institution_id: 'alpha',
        subject_id: 'Biology', subject_label: 'Biology', lab_name: 'Instrument Room',
        url: '/real-labs?scene=original' }],
    },
  };
  const sandbox = {
    document: { getElementById: element }, URL, fixture,
    Option: function (text, value) { this.text = text; this.value = value; },
    fetch: () => new Promise(() => {}),
    location: { origin: 'http://localhost', assign(url) { this.assigned = url; } },
  };
  vm.createContext(sandbox);
  vm.runInContext(script, sandbox);
  const run = command => vm.runInContext(command, sandbox);
  run('CATALOG = fixture; populateUniUniversities(); populateUniSubjects();');
  return {
    sandbox, element, run,
    options: id => element(id).children.map(option => option.text),
    change(id, value) {
      element(id).value = value;
      for (const listener of element(id).listeners.change || []) listener();
    },
  };
}

test('both collections share sorted schools and preserve instrument links', async () => {
  const view = picker();
  assert.deepEqual(view.options('uni-university'), ['All universities', 'Alpha University', 'Empty University', 'Zeta University']);
  view.change('uni-university', 'alpha');
  assert.deepEqual(view.options('uni-subject'), ['Biology', 'Chemistry']);
  assert.deepEqual(view.options('uni-lab'), ['Instrument Room']);
  assert.equal(view.element('uni-generate').disabled, false);
  await view.run('uniGenerateScene()');
  assert.equal(view.sandbox.location.assigned, 'http://localhost/real-labs?scene=original');
  view.change('uni-subject', 'Chemistry');
  assert.deepEqual(view.options('uni-lab'), ['Optics Room']);
  assert.match(view.element('uni-preview').innerHTML, /<video.*real_lab_shared_display\.mp4/);
});

test('display selection plays its clip and generates the exact native task', async () => {
  const view = picker();
  view.change('uni-university', 'real_lab:Zeta University');
  assert.equal(view.element('uni-lab').value, 'real_lab_zeta_display');
  assert.equal(view.element('uni-generate').disabled, false);
  assert.match(view.element('uni-preview').innerHTML, /autoplay loop muted playsinline controls/);
  let request;
  view.sandbox.fetch = async (url, options) => {
    request = { url, body: JSON.parse(options.body) };
    return { ok: true, json: async () => ({ image_png_base64: 'fresh', task_info: { prefix: 'Preview' }, robot: { display_name: 'UR5e' } }) };
  };
  await view.run('uniGenerateScene()');
  assert.equal(request.url, '/api/scene');
  assert.equal(request.body.task, 'real_lab_zeta_display');
  assert.equal(request.body.robot, 'ur5e');
  assert.equal(view.sandbox.location.assigned, undefined);
  assert.match(view.element('uni-preview').innerHTML, /base64,fresh/);
});

test('late display render cannot replace a newly selected instrument scene or clip', async () => {
  const view = picker();
  view.change('uni-university', 'real_lab:Zeta University');
  let finish;
  view.sandbox.fetch = () => new Promise(resolve => { finish = resolve; });
  const pending = view.run('uniGenerateScene()');
  view.change('uni-university', 'alpha');
  view.change('uni-subject', 'Chemistry');
  const clip = view.element('uni-preview').innerHTML;
  finish({ ok: true, json: async () => ({ image_png_base64: 'stale', task_info: {}, robot: {} }) });
  await pending;
  assert.equal(view.element('uni-preview').innerHTML, clip);
  view.change('uni-university', 'empty');
  assert.equal(view.element('uni-generate').disabled, true);
  assert.equal(view.element('uni-preview').innerHTML, '');
});

test('display-only catalogues work without the instrument directory and do not duplicate handlers', () => {
  const view = picker();
  delete view.sandbox.fixture.university;
  const before = JSON.stringify(view.sandbox.fixture);
  for (let count = 0; count < 3; count++) {
    view.run('populateUniUniversities(); populateUniSubjects();');
    view.change('uni-university', 'real_lab:Zeta University');
  }
  assert.deepEqual(view.options('uni-university'), ['All universities', 'Alpha University', 'Zeta University']);
  assert.equal(view.run('currentUniTask().name'), 'real_lab_zeta_display');
  assert.equal(JSON.stringify(view.sandbox.fixture), before);
  for (const id of ['uni-university', 'uni-subject', 'uni-lab']) {
    assert.equal(view.element(id).listeners.change.length, 1);
  }
});

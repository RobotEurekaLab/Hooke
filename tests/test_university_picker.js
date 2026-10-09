// Run with: node --test tests/test_university_picker.js
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const html = readFileSync(resolve(__dirname, '../Hooke/webui/static/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function picker(directory) {
  const elements = new Map();
  const document = {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, {
        value: '', textContent: '', innerHTML: '', disabled: false,
        style: {}, classList: { toggle() {} }, children: [], listeners: {},
        replaceChildren(...children) {
          this.children = children;
          this.value = children[0]?.value || '';
          this.innerHTML = '';
        },
        add(option) { this.children.push(option); },
        addEventListener(event, listener) {
          (this.listeners[event] ||= []).push(listener);
        },
      });
      return elements.get(id);
    },
  };
  const sandbox = {
    document, URL,
    Option: function (text, value) { this.text = text; this.value = value; },
    fetch: () => new Promise(() => {}),
    location: { origin: 'http://localhost', assign(url) { this.assigned = url; } },
    fixture: {
      tasks: ['alpha_a', 'alpha_z', 'beta', 'nasa', 'generic'].map(name => ({
        name, category: 'biology', description: `${name} description`, robot: 'missing',
      })),
      university: directory,
    },
  };
  vm.createContext(sandbox);
  vm.runInContext(script, sandbox);
  vm.runInContext('CATALOG = fixture; populateUniUniversities(); populateUniSubjects();', sandbox);
  return {
    sandbox,
    element: id => document.getElementById(id),
    run: command => vm.runInContext(command, sandbox),
    change(id, value) {
      const element = document.getElementById(id);
      element.value = value;
      for (const listener of element.listeners.change || []) listener();
    },
    options: id => document.getElementById(id).children.map(option => option.text),
  };
}

function directory() {
  return {
    ranking: { title: 'QS World University Rankings', edition: 2027 },
    universities: [
      { id: 'zeta', name: 'Zeta University' },
      { id: 'alpha', name: 'Alpha University' },
      { id: 'empty', name: 'Empty University', candidate_count: 2 },
    ],
    labs: [
      { scene_id: 'alpha_z', task_name: 'alpha_z', institution_id: 'alpha',
        subject_id: 'zoology', subject_label: 'Zoology', lab_name: 'Zeta lab', url: '/real-labs?scene=alpha_z' },
      { scene_id: 'beta', task_name: 'beta', institution_id: 'zeta',
        subject_id: 'biology', subject_label: 'Biology', lab_name: 'Beta lab', url: '/real-labs?scene=beta' },
      { scene_id: 'alpha_a', task_name: 'alpha_a', institution_id: 'alpha',
        subject_id: 'biology', subject_label: 'Biology', lab_name: 'Alpha lab', url: '/real-labs?scene=alpha_a' },
      { scene_id: 'nasa', task_name: 'nasa', institution_id: 'nasa',
        subject_id: 'astronomy', subject_label: 'Astronomy', lab_name: 'NASA lab' },
      { scene_id: 'unbuilt', task_name: 'not_a_task', institution_id: 'alpha',
        subject_id: 'astronomy', subject_label: 'Astronomy', lab_name: 'Unbuilt candidate' },
    ],
  };
}

test('directory sorts actual universities, subjects and labs, excluding unbuilt and nonuniversity records', () => {
  const view = picker(directory());
  assert.deepEqual(view.options('uni-university'), ['All universities', 'Alpha University', 'Empty University', 'Zeta University']);
  assert.deepEqual(view.options('uni-subject'), ['Biology', 'Zoology']);
  assert.deepEqual(view.options('uni-lab'), ['Alpha lab', 'Beta lab']);
  assert.match(view.element('uni-scope-hint').textContent, /2027, ranks 1–100/);
});

test('school selection filters both subject and lab by institution ID and opens its own scene', async () => {
  const view = picker(directory());
  view.change('uni-university', 'zeta');
  assert.deepEqual(view.options('uni-subject'), ['Biology']);
  assert.deepEqual(view.options('uni-lab'), ['Beta lab']);
  assert.equal(view.element('uni-generate').disabled, false);
  await view.run('uniGenerateScene()');
  assert.equal(view.sandbox.location.assigned, 'http://localhost/real-labs?scene=beta');
  view.change('uni-university', 'alpha');
  assert.deepEqual(view.options('uni-subject'), ['Biology', 'Zoology']);
  view.change('uni-subject', 'zoology');
  assert.deepEqual(view.options('uni-lab'), ['Zeta lab']);
});

test('multiple scenes in one laboratory have distinct sorted labels and correct destinations', async () => {
  const data = directory();
  Object.assign(data.labs[0], {
    subject_id: 'biology', subject_label: 'Biology', lab_name: 'Alpha lab',
    lab_id: 'alpha_lab', title: 'Zeta room',
  });
  Object.assign(data.labs[2], { lab_id: 'alpha_lab', title: 'Alpha room' });
  const view = picker(data);
  view.change('uni-university', 'alpha');
  assert.deepEqual(view.options('uni-lab'), ['Alpha lab — Alpha room', 'Alpha lab — Zeta room']);
  await view.run('uniGenerateScene()');
  assert.equal(view.sandbox.location.assigned, 'http://localhost/real-labs?scene=alpha_a');
  view.change('uni-lab', 'alpha_z');
  await view.run('uniGenerateScene()');
  assert.equal(view.sandbox.location.assigned, 'http://localhost/real-labs?scene=alpha_z');
});

test('empty schools clear stale previews and disable launch without falling back to another school', async () => {
  const view = picker(directory());
  view.element('uni-preview').innerHTML = '<img src="stale.png">';
  view.change('uni-university', 'empty');
  for (const id of ['uni-subject', 'uni-lab', 'uni-generate']) assert.equal(view.element(id).disabled, true);
  assert.equal(view.element('uni-preview').innerHTML, '');
  assert.match(view.element('uni-robot-info').textContent, /No reconstructed labs.*this university/);
  await view.run('uniGenerateScene()');
  assert.equal(view.sandbox.location.assigned, undefined);
});

test('missing and empty directory data stay unavailable; absent robot metadata is tolerated', () => {
  for (const data of [undefined, null, {}, { universities: null, labs: null }]) {
    const view = picker(data);
    assert.deepEqual(view.options('uni-university'), ['All universities']);
    assert.equal(view.element('uni-generate').disabled, true);
    assert.match(view.element('uni-robot-info').textContent, /No reconstructed university labs/);
  }
  const view = picker(directory());
  assert.equal(view.element('uni-robot-info').textContent, 'alpha_a description');
});

test('repopulating dependent dropdowns does not register duplicate change handlers', () => {
  const view = picker(directory());
  for (let count = 0; count < 4; count++) {
    view.change('uni-university', 'alpha');
    view.change('uni-subject', 'zoology');
    view.change('uni-university', '');
    view.run('populateUniUniversities(); populateUniSubjects();');
  }
  for (const id of ['uni-university', 'uni-subject', 'uni-lab']) {
    assert.equal(view.element(id).listeners.change.length, 1);
  }
});

test('a preview response for a previously selected lab cannot overwrite the new selection', async () => {
  const data = directory();
  data.labs[2].url = null;
  const view = picker(data);
  let finish;
  view.sandbox.fetch = () => new Promise(resolve => { finish = resolve; });
  const pending = view.run('uniGenerateScene()');
  view.change('uni-university', 'empty');
  finish({ ok: true, json: async () => ({ image_png_base64: 'old', task_info: { prefix: 'old' }, robot: { display_name: 'old' } }) });
  await pending;
  assert.equal(view.element('uni-preview').innerHTML, '');
});

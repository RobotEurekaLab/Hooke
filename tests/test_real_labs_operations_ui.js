// Run with: node --test tests/test_real_labs_operations_ui.js
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {resolve} = require('node:path');
const {test} = require('node:test');
const vm = require('node:vm');

const html = readFileSync(resolve(__dirname, '../Hooke/webui/static/real_labs.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function view() {
  const nodes = new Map(), created = [];
  function node(tag='div') {
    const item = {tag, value:'', textContent:'', children:[], disabled:false, hidden:false,
      append(...children) {this.children.push(...children); if (this.tag === 'select' && !this.value) this.value = children[0]?.value || '';},
      replaceChildren(...children) {this.children=children; if (this.tag === 'select') this.value=children[0]?.value || '';},
      setAttribute() {}, checkValidity() {return true;}, reportValidity() {},
    };
    created.push(item); return item;
  }
  const document = {
    getElementById(id) {
      if (!nodes.has(id)) nodes.set(id,node(['scene','camera','equipment'].includes(id)?'select':'div'));
      return nodes.get(id);
    },
    createElement:node,
    querySelectorAll() {return created;},
  };
  const context = {document, URL, URLSearchParams, location:{search:''},
    fetch:()=>new Promise(()=>{}), setTimeout:fn=>fn(),
  };
  vm.createContext(context); vm.runInContext(script,context);
  document.getElementById('scene').value='rochester_genomics';
  vm.runInContext('busy(false)',context);
  return {context, get:id=>document.getElementById(id), run:code=>vm.runInContext(code,context)};
}

function scene() {
  const control = {measured:0,target:0,range:[-0.7,0],units:'m'};
  return {
    time_s:0,manifest:{equipment:[{id:'liquid_handler',kind:'liquid_handler'}],source_urls:[]},
    equipment:[{id:'liquid_handler',controls:{loader_x:control,loader_z:{...control,range:[0,0.105]}},capabilities:[]}],
    operations:[{id:'plate_loading',equipment_id:'liquid_handler',label:'Load and retrieve plate',description:'Run a complete contact operation.'}],
    control_policy:[{equipment_id:'liquid_handler',control:'loader_x',mode:'operation_only',operation:'plate_loading'}],
  };
}

function response(data,status=200) {return {ok:status<400,status,json:async()=>data};}

test('operation-only axes have no Apply control, while available axes and operation buttons remain',()=>{
  const ui=view();ui.context.fixture=scene();ui.run('showState(fixture)');
  const rows=ui.get('controls').children;
  assert.equal(rows.length,2);
  assert.match(rows[0].children[1].textContent,/"Load and retrieve plate" operation for this control/);
  assert.equal(rows[0].children.some(n=>n.tag==='input'),false);
  assert.equal(rows[1].children.some(n=>n.tag==='input'),true);
  assert.equal(ui.get('operations').children[1].children[0].textContent,'Load and retrieve plate');
});

test('dry command range is separate from full physical range',()=>{
  const ui=view(), fixture=scene();
  fixture.control_policy=[{equipment_id:'liquid_handler',control:'loader_z',mode:'bounded',lower:0,upper:0.012}];
  fixture.equipment[0].controls.loader_z.target=0.024;
  ui.context.fixture=fixture;ui.run('showState(fixture)');
  const row=ui.get('controls').children[1];
  assert.equal(row.children[2].max,0.012);
  assert.equal(row.children[3].children[0].value,0.012);
  assert.equal(fixture.equipment[0].controls.loader_z.range[1],0.105);
  assert.match(row.children[1].textContent,/dry inspection/);
});

test('operation starts once, polls through running, then displays measured terminal state',async()=>{
  const ui=view(), next=scene();next.time_s=111.6;
  const calls=[], job={id:'a',operation:'plate_loading',status:'running'};
  const replies=[{job},{job},{job:{...job,status:'succeeded',result:{success:true}},state:next}];
  ui.context.fetch=async(path,options)=>{calls.push([path,options]);return response(replies.shift());};
  await ui.run("startOperation('plate_loading')");
  assert.equal(calls.length,3);
  assert.equal(calls[0][1].method,'POST');
  assert.equal(calls[1][1].method,undefined);
  assert.match(ui.get('result').textContent,/111.6/);
  assert.equal(ui.get('status').textContent,'Operation completed');
  assert.equal(ui.run('loading'),false);
  assert.equal(ui.run('pendingOperation'),null);
});

test('measured failure remains failure and never issues reset',async()=>{
  const ui=view(), job={id:'a',operation:'plate_loading',status:'failed',error:'sample_not_supported'};
  const calls=[];
  ui.context.fetch=async(path)=>{calls.push(path);return response({job,state:scene()});};
  await ui.run("watchOperation({id:'a',operation:'plate_loading',status:'running'})");
  assert.equal(ui.get('status').textContent,'Operation failed');
  assert.equal(ui.get('error').textContent,'sample_not_supported');
  assert.equal(calls.some(path=>path.includes('reset')),false);
});

test('lost connection preserves pending job and offers explicit status retry',async()=>{
  const ui=view();
  ui.context.fetch=async()=>{throw new Error('Connection lost');};
  await ui.run("watchOperation({id:'a',operation:'plate_loading',status:'running'})");
  assert.equal(ui.run('loading'),true);
  assert.equal(ui.get('retry-operation').hidden,false);
  assert.equal(ui.get('retry-operation').disabled,false);
  ui.context.fetch=async()=>response({job:{id:'a',operation:'plate_loading',status:'succeeded'},state:scene()});
  await ui.get('retry-operation').onclick();
  // The event handler starts a promise; wait for its microtasks to finish.
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(ui.run('pendingOperation'),null);
  assert.equal(ui.run('loading'),false);
  assert.equal(ui.get('retry-operation').hidden,true);
});

test('reload of a busy scene resumes polling its existing job without starting another',async()=>{
  const ui=view(), job={id:'existing',operation:'plate_loading',status:'running'}, calls=[];
  ui.context.fetch=async(path,options)=>{
    calls.push([path,options]);
    return path.endsWith('/state')?response({error:'Scene busy',job},409):response({job:{...job,status:'succeeded'},state:scene()});
  };
  await ui.run('loadScene()');
  assert.equal(ui.get('status').textContent,'Operation completed');
  assert.equal(calls.length,2);
  assert.equal(calls.some(([,options])=>options.method==='POST'),false);
});

test('lost POST response recovers server job without submitting the operation twice',async()=>{
  const ui=view(), calls=[];
  ui.context.fetch=async(path,options)=>{
    calls.push([path,options]);
    if(options.method==='POST')throw new Error('Response lost after request accepted');
    return response({job:{id:'accepted',operation:'plate_loading',status:'succeeded'},state:scene()});
  };
  await ui.run("startOperation('plate_loading')");
  assert.equal(calls.filter(([,options])=>options.method==='POST').length,1);
  assert.equal(ui.get('status').textContent,'Operation completed');
});

test('terminal state fetch can be retried without losing the job',async()=>{
  const ui=view();
  ui.context.fetch=async(path)=>{
    if(path.endsWith('/state'))throw new Error('State unavailable');
    return response({job:{id:'done',operation:'plate_loading',status:'succeeded'}});
  };
  await ui.run("watchOperation({id:'done',operation:'plate_loading',status:'running'})");
  assert.equal(ui.run('pendingOperation.id'),'done');
  ui.context.fetch=async()=>response({job:{id:'done',operation:'plate_loading',status:'succeeded'},state:scene()});
  await ui.get('retry-operation').onclick();
  assert.equal(ui.run('loading'),false);
});

test('a replacement job is observed rather than causing an endless stale-job retry',async()=>{
  const ui=view();
  ui.context.fetch=async()=>response({job:{id:'new',operation:'retract',status:'succeeded'},state:scene()});
  await ui.run("watchOperation({id:'old',operation:'plate_loading',status:'running'})");
  assert.match(ui.get('result').textContent,/retract completed/);
  assert.equal(ui.run('loading'),false);
});

test('a late scene error cannot attach its job to the newly selected scene',async()=>{
  const ui=view();let finish;
  ui.context.fetch=()=>new Promise(resolve=>{finish=resolve;});
  const loading=ui.run('loadScene()');
  ui.run('selectionVersion++');ui.get('scene').value='leeds_environmental_tribology';
  finish(response({error:'busy',job:{id:'old',operation:'plate_loading',status:'running'}},409));
  await loading;
  assert.equal(ui.run('pendingOperation'),null);
});

test('nonfinite physics is shown as unavailable without offering further motion',()=>{
  const ui=view(), fixture=scene();
  Object.assign(fixture,{time_s:null,invalid_physics_state:true,state_error:'Invalid measured state; reset required.'});
  ui.context.fixture=fixture;ui.run('showState(fixture)');
  assert.equal(ui.get('operations').children.length,0);
  assert.equal(ui.get('controls').children.length,1);
  assert.match(ui.get('controls').children[0].textContent,/Invalid measured state/);
  assert.equal(ui.get('time').textContent,'Unavailable');
  assert.equal(ui.run('fmt(null)'),'unavailable');
});

test('completion time is historical while the displayed scene state stays current',async()=>{
  const ui=view();
  ui.context.fetch=async()=>response({job:{id:'done',operation:'plate_loading',status:'succeeded'},state:scene(),completed_state:{time_s:111.6}});
  await ui.run("watchOperation({id:'done',operation:'plate_loading',status:'running'})");
  assert.match(ui.get('result').textContent,/111.6/);
  assert.equal(ui.get('time').textContent,'0.00 s');
});

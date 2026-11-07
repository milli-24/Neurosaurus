// Game rules, kept apart from drawing so they can be checked without a browser.
// Constants must match experiments/neurodino.py.
const SIM = { GRAVITY: 2600, JUMP_V: 820, TICK_S: 0.05, DINO_W: 34, CACTUS_W: 22, CACTUS_H: 44,
              LEAD_IN: 170, KEY_START: 470, EXIT_X: -100 };

function createSim(trials, events = {}) {
  const s = { y: 0, vy: 0, cactus: null, armed: false, clock: 0, tick: 0, trial: 0, cleared: 0, hit: 0,
              spikes: new Array(64).fill(0), fired: 0, decoder: 'network' };

  function spawn() {
    const t = trials && trials[s.trial % trials.length];
    const dist = t ? t.start_dist + SIM.LEAD_IN : SIM.KEY_START;
    s.cactus = { x: SIM.DINO_W + dist, speed: t ? t.speed : 120 + Math.random() * 360, scored: false };
    s.armed = false; s.clock = 0; s.tick = 0;
  }

  s.jump = () => {
    if (s.y > 0) return false;
    s.vy = SIM.JUMP_V;
    if (events.onJump) events.onJump();
    return true;
  };

  function control() {
    const t = trials[s.trial % trials.length];
    const row = t.spikes[Math.min(s.tick, t.spikes.length - 1)];
    s.spikes = row;
    s.fired += row.reduce((a, b) => a + b, 0);
    if (t.jump_tick[s.decoder] === s.tick) s.jump();
    s.tick++;
  }

  s.step = dt => {
    const c = s.cactus;
    if (trials) {
      const t = trials[s.trial % trials.length];
      if (!s.armed && c.x - SIM.DINO_W <= t.start_dist) { s.armed = true; s.clock = 0; control(); }
      else if (s.armed) {
        s.clock += dt;
        while (s.clock >= SIM.TICK_S) { s.clock -= SIM.TICK_S; control(); }
      }
    }
    c.x -= c.speed * dt;
    s.vy -= SIM.GRAVITY * dt;
    s.y = Math.max(0, s.y + s.vy * dt);
    if (s.y === 0 && s.vy < 0) s.vy = 0;
    const overlap = c.x < SIM.DINO_W && c.x + SIM.CACTUS_W > 0 && s.y < SIM.CACTUS_H;
    if (overlap && !c.scored) { c.scored = true; s.hit++; if (events.onHit) events.onHit(); }
    if (c.x + SIM.CACTUS_W < 0 && !c.scored) { c.scored = true; s.cleared++; if (events.onClear) events.onClear(); }
    if (c.x + SIM.CACTUS_W < SIM.EXIT_X) { s.trial++; spawn(); }
  };

  spawn();
  return s;
}

if (typeof module !== 'undefined') module.exports = { SIM, createSim };

"""Visual-only lab-equipment scenes built via `static_display.py`'s generic
archetype -- see that module's docstring for the scope tradeoff (real
physics/collision, no scripted-expert interaction). Each item here is
just a `StaticDisplaySpec` + a `<name>Task`/`<name>Expert` pair generated
from it, so `task_catalog.py`'s `CatalogEntry(module="archetypes.static_items",
cls="...")` can resolve it the same way it resolves a real hand-written
task module.
"""
from archetypes.static_display import StaticDisplaySpec, make_static_task

AnalyticalBalanceTask, AnalyticalBalanceExpert = make_static_task(StaticDisplaySpec(
    name="analytical_balance_display",
    scene_file="mani_analytical_balance.xml",
    prompt="an analytical balance on the bench",
))

HotPlateStirrerTask, HotPlateStirrerExpert = make_static_task(StaticDisplaySpec(
    name="hot_plate_stirrer_display",
    scene_file="mani_hot_plate_stirrer.xml",
    prompt="a magnetic hot plate stirrer on the bench",
))

RoundBottomFlaskStandTask, RoundBottomFlaskStandExpert = make_static_task(StaticDisplaySpec(
    name="round_bottom_flask_stand_display",
    scene_file="mani_round_bottom_flask_stand.xml",
    prompt="a round-bottom flask on a ring stand",
))

TabletPressTask, TabletPressExpert = make_static_task(StaticDisplaySpec(
    name="tablet_press_display",
    scene_file="mani_tablet_press.xml",
    prompt="a pharmaceutical tablet press",
))

LaminarFlowHoodTask, LaminarFlowHoodExpert = make_static_task(StaticDisplaySpec(
    name="laminar_flow_hood_display",
    scene_file="mani_laminar_flow_hood.xml",
    prompt="a laminar flow hood for sterile pharmaceutical work",
))

AutoclaveTask, AutoclaveExpert = make_static_task(StaticDisplaySpec(
    name="autoclave_display",
    scene_file="mani_autoclave.xml",
    prompt="a benchtop autoclave sterilizer",
))

GloveboxTask, GloveboxExpert = make_static_task(StaticDisplaySpec(
    name="glovebox_display",
    scene_file="mani_glovebox.xml",
    prompt="an inert-atmosphere glovebox for battery cell assembly",
))

CoinCellCrimperTask, CoinCellCrimperExpert = make_static_task(StaticDisplaySpec(
    name="coin_cell_crimper_display",
    scene_file="mani_coin_cell_crimper.xml",
    prompt="a manual coin cell crimper for battery assembly",
))

#!/usr/bin/env python3

from functools import lru_cache
from multiprocessing.pool import Pool
import itertools as it  # for cartesian product
import time
import random
import os
import logging
import argparse
import numpy as np
# import matplotlib.pyplot as plt

from src.config import SimulationConfig
from .domain import Domain

#######################################################
MAX_STEPS = SimulationConfig.simulator.time_limit
steps = range(MAX_STEPS)

cellSize = 0.4  # m
vmax = 1.2
dt = cellSize / vmax  # time step

from_x, to_x = 1, 63  # todo parse this too
from_y, to_y = 1, 63  # todo parse this too
DEFAULT_BOX = [from_x, to_x, from_y, to_y]
del from_x, to_x, from_y, to_y


# DFF = np.ones( (dim_x, dim_y) ) # dynamic floor field
#######################################################

logfile = "log.dat"
logging.basicConfig(
    filename=logfile,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def check_N_pedestrians(_box, N_pedestrians):
    """
    check if <N_pedestrian> is too big. if so change it to fit in <box>
    """
    # holding box, where to distribute pedestrians
    # ---------------------------------------------------
    _from_x = _box[0]
    _to_x = _box[1]
    _from_y = _box[2]
    _to_y = _box[3]
    # ---------------------------------------------------
    nx = _to_x - _from_x + 1
    ny = _to_y - _from_y + 1
    if N_pedestrians > nx * ny:
        logging.warning(
            "N_pedestrians (%d) is too large (max. %d). Set to max."
            % (N_pedestrians, nx * ny)
        )
        N_pedestrians = nx * ny

    return N_pedestrians


def init_DFF():
    """ """
    return np.zeros((dim_x, dim_y))


def update_DFF(dff, diff):
    # for cell in diff:
    #    assert walls[cell] > -10
    #     dff[cell] += 1

    dff += diff

    for i, j in it.chain(
        it.product(range(1, dim_x - 1), range(1, dim_y - 1)), exit_cells
    ):
        for _ in range(int(dff[i, j])):
            if np.random.rand() < delta:  # decay
                dff[i, j] -= 1
            elif np.random.rand() < alpha:  # diffusion
                dff[i, j] -= 1
                dff[random.choice(get_neighbors((i, j)))] += 1
        assert walls[i, j] > -10 or dff[i, j] == 0, (dff, i, j)
    # dff[:] = np.ones((dim_x, dim_y))


@lru_cache(1)
def init_SFF(_exit_cells, _dim_x, _dim_y, drawS):
    # start with exit's cells
    SFF = np.empty((_dim_x, _dim_y))  # static floor field
    SFF[:] = np.sqrt(_dim_x**2 + _dim_y**2)

    make_videos = 0

    cells_initialised = []
    for e in _exit_cells:
        cells_initialised.append(e)
        SFF[e] = 0

    while cells_initialised:
        cell = cells_initialised.pop(0)
        neighbor_cells = get_neighbors(cell)
        for neighbor in neighbor_cells:
            # print ("cell",cell, "neighbor",neighbor)
            if SFF[cell] + 1 < SFF[neighbor]:
                SFF[neighbor] = SFF[cell] + 1
                cells_initialised.append(neighbor)
                # print(SFF)
        # print(cells_initialised)

    return SFF


@lru_cache(16 * 1024)
def get_neighbors(cell):
    """
    von Neumann neighborhood
    """
    neighbors = []
    i, j = cell
    if i < dim_x - 1 and walls[(i + 1, j)] >= 0:
        neighbors.append((i + 1, j))
    if i >= 1 and walls[(i - 1, j)] >= 0:
        neighbors.append((i - 1, j))
    if j < dim_y - 1 and walls[(i, j + 1)] >= 0:
        neighbors.append((i, j + 1))
    if j >= 1 and walls[(i, j - 1)] >= 0:
        neighbors.append((i, j - 1))

    # moore
    if moore:
        if i >= 1 and j >= 1 and walls[(i - 1, j - 1)] >= 0:
            neighbors.append((i - 1, j - 1))
        if i < dim_x - 1 and j < dim_y - 1 and walls[(i + 1, j + 1)] >= 0:
            neighbors.append((i + 1, j + 1))
        if i < dim_x - 1 and j >= 1 and walls[(i + 1, j - 1)] >= 0:
            neighbors.append((i + 1, j - 1))
        if i >= 1 and j < dim_y - 1 and walls[(i - 1, j + 1)] >= 0:
            neighbors.append((i - 1, j + 1))

    # not shuffling singnificantly alters the simulation...
    random.shuffle(neighbors)
    return neighbors


def seq_update_cells(domain, sff, dff, kappaD, kappaS, shuffle, reverse):
    """
    sequential update
    input
       - domain:
       - sff:
       - dff:
       - prob_walls:
       - kappaD:
       - kappaS:
       - rand: random shuffle
    return
       - new domain.peds
    """

    tmp_peds = np.empty_like(domain.peds)  # temporary cells
    np.copyto(tmp_peds, domain.peds)

    dff_diff = np.zeros((dim_x, dim_y))

    grid = list(it.product(range(1, dim_x - 1), range(1, dim_y - 1))) + list(exit_cells)
    if shuffle:  # sequential random update
        random.shuffle(grid)
    elif reverse:  # reversed sequential update
        grid.reverse()

    for i, j in grid:  # walk through all cells in geometry
        if domain.peds[i, j] == 0:
            continue

        if (i, j) in exit_cells:
            tmp_peds[i, j] = 0
            dff_diff[i, j] += 1
            continue

        p = 0
        probs = {}
        cell = (i, j)
        for neighbor in get_neighbors(cell):  # get the sum of probabilities
            # original code:
            # probability = np.exp(-kappaS * sff[neighbor]) * np.exp(kappaD * dff[neighbor]) * \
            # (1 - tmp_peds[neighbor])
            # the absolute value of the exponents can get very large yielding 0 or
            # inifite probability.
            # to prevent this we multiply every probability with exp(kappaS * sff[cell) and
            # exp(-kappaD * dff[cell]).
            # since the probabilities are normalized this doesn't have any effect on the model

            probability = (
                np.exp(kappaS * (sff[cell] - sff[neighbor]))
                * np.exp(kappaD * (dff[neighbor] - dff[cell]))
                * (1 - tmp_peds[neighbor])
            )

            p += probability
            probs[neighbor] = probability

        if p == 0:  # pedestrian in cell can not move
            continue

        r = np.random.rand() * p
        # print ("start update")
        for neighbor in get_neighbors(cell):  # TODO: shuffle?
            r -= probs[neighbor]
            if r <= 0:  # move to neighbor cell
                tmp_peds[neighbor] = 1
                tmp_peds[i, j] = 0
                dff_diff[i, j] += 1

                domain.move_ped(cell, neighbor)
                break

    return tmp_peds, dff_diff


def print_logs(N_pedestrians, width, height, t, dt, nruns, Dt):
    """
    print some infos to the screen
    """
    # print(' << --------------------------------------------------- ')
    # print("Simulation of %d pedestrians" % N_pedestrians)
    # print("Simulation space (%.2f x %.2f) m^2" % (width, height))
    # print("SFF:  %.2f | DFF: %.2f" % (kappaS, kappaD))
    # print("Mean Evacuation time: %.2f s, runs: %d" % (t * dt / nruns, nruns))
    # print("Total Run time: %.2f s" % Dt)
    # print("Factor: x%.2f" % (dt * t / Dt))
    # print('  --------------------------------------------------- >> ')
    pass


def setup_dir(dir, clean):
    print("make ", dir)
    if os.path.exists(dir) and clean:
        os.system("rm -rf %s" % dir)
    os.makedirs(dir, exist_ok=True)


def simulate(args, domain, window):
    n, npeds, box, sff, shuffle, reverse, drawP, giveD = args
    # print(
    #     "init %d agents in box=[%d, %d, %d, %d]"
    #     % (npeds, box[0], box[1], box[2], box[3])
    # )

    dff = init_DFF()

    old_dffs = []
    for t in steps:  # simulation loop
        if window:
            window.updateGrid()
            time.sleep(0.1)
        # print(
        #     "\tn: %3d ----  t: %3d |  N: %3d"
        #     % (n, t, int(domain.get_left_pedestrians_count()))
        # )

        domain.peds, dff_diff = seq_update_cells(
            domain, sff, dff, kappaD, kappaS, shuffle, reverse
        )

        update_DFF(dff, dff_diff)
        if giveD:
            old_dffs.append((t, dff.copy()))

        domain.increase_pedestrians_t_star()

        if (
            domain.get_left_pedestrians_count() == 0
        ):  # is everybody out? TODO: check this. Some bug is lurking here
            print("Quite simulation")
            break
    # else:
    #     raise TimeoutError("simulation taking too long")

    if giveD:
        return t, old_dffs
    else:
        return t


def check_box(box):
    """
    exit if box is not well defined
    """
    assert box[0] < box[1], "from_x smaller than to_x"
    assert box[2] < box[3], "from_y smaller than to_y"


def main(domain: Domain, window=None):
    global kappaS, kappaD, dim_y, dim_x, exit_cells, SFF, alpha, delta, walls, parallel, box, moore
    # init parameters

    kappaS = SimulationConfig.simulator.cellular_automaton_parameters.kappa_static
    kappaD = SimulationConfig.simulator.cellular_automaton_parameters.kappa_dynamic
    npeds = random.uniform(
        SimulationConfig.num_pedestrians[0], SimulationConfig.num_pedestrians[1]
    )
    moore = (
        SimulationConfig.simulator.cellular_automaton_parameters.neighborhood == "Moore"
    )
    nruns = SimulationConfig.simulator.cellular_automaton_parameters.num_runs

    drawS = SimulationConfig.simulator.cellular_automaton_parameters.plotS
    drawP = SimulationConfig.simulator.cellular_automaton_parameters.plotP
    shuffle = SimulationConfig.simulator.cellular_automaton_parameters.shuffle
    reverse = SimulationConfig.simulator.cellular_automaton_parameters.reverse
    drawD = SimulationConfig.simulator.cellular_automaton_parameters.plotD
    drawD_avg = SimulationConfig.simulator.cellular_automaton_parameters.plotAvgD
    clean_dirs = SimulationConfig.simulator.cellular_automaton_parameters.clean
    width = domain.width  # in meters
    height = domain.height  # in meters
    parallel = SimulationConfig.simulator.cellular_automaton_parameters.parallel
    box = DEFAULT_BOX
    check_box(box)

    if parallel and drawP:
        raise NotImplementedError("cannot plot pedestrians when multiprocessing")

    # TODO check if width and hight are multiples of cellSize
    dim_y = height  # number of columns, add ghost cells
    dim_x = width  # number of rows, add ghost cells
    # print(" dim_x: ", dim_x, " dim_y: ", dim_y)
    if box == DEFAULT_BOX:
        # print("box == room")
        box = [1, dim_x - 2, 1, dim_y - 2]

    alpha = SimulationConfig.simulator.cellular_automaton_parameters.diffusion
    delta = SimulationConfig.simulator.cellular_automaton_parameters.decay

    exit_cells = frozenset(domain.get_exit_cells())

    npeds = check_N_pedestrians(box, npeds)

    walls = domain.walls

    sff = init_SFF(exit_cells, dim_x, dim_y, drawS)


    t1 = time.time()
    tsim = 0

    if drawP:
        setup_dir("peds", clean_dirs)
    if drawD or drawD_avg:
        setup_dir("dff", clean_dirs)

    times = []
    old_dffs = []

    for n in range(nruns):
        # print("n= ", n, " nruns=", nruns)
        if drawD_avg or drawD:
            t, dffs = simulate(
                (n, npeds, box, sff, shuffle, reverse, drawP, drawD_avg or drawD),
                domain,
                window,
            )
            old_dffs += dffs
        else:
            t = simulate(
                (n, npeds, box, sff, shuffle, reverse, drawP, drawD_avg or drawD),
                domain,
                window,
            )
        tsim += t
        # print("time ", tsim)
        times.append(t * dt)
    # if moore:
    #     # print("save moore.npy")
    #     np.save("moore.npy", times)
    # else:
    #     # print("save neumann.npy")
    #     np.save("neumann.npy", times)

    t2 = time.time()
    print_logs(npeds, width, height, tsim, dt, nruns, t2 - t1)
    if drawD_avg:
        print("plotting average DFF")
        if moore:
            title = "DFF-avg_Moore_runs_%d_N%d_S%.2f_D%.2f" % (
                nruns,
                npeds,
                kappaS,
                kappaD,
            )
        else:
            title = "DFF-avg_Neumann_runs_%d_N%d_S%.2f_D%.2f" % (
                nruns,
                npeds,
                kappaS,
                kappaD,
            )
    # title=r"$t = {:.2f}$ s, N={}, #runs = {}, $\kappa_S={}\;, \kappa_D={}$".format(sum(times), npeds, nruns, kappaS, kappaD)
    if drawD:
        print("plotting DFFs...")
        max_dff = max(field.max() for _, field in old_dffs)
        for tm, dff in old_dffs:
            print("t: %3.4d" % tm)

    return times


if __name__ == "__main__":
    main()

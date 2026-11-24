import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'sim'))
import env as EM

CITY = os.environ.get('TRAJ_CITY', 'city21'); K = int(os.environ.get('TRAJ_K', '6'))


def load(platform):
    c = EM.City(os.path.join(os.environ['TRAJ_PRELIM'], f'{"g" if platform == "ground" else ""}{CITY}.pkl'))
    c.name = CITY
    m = EM.make_mission(c, K)
    return c, m

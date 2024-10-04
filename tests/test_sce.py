"""
Contains unit tests for sce
"""

# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long
# pylint: disable=R0801
# pylint: disable=invalid-name
# pylint: disable=wildcard-import
# pylint: disable=unused-wildcard-import

import os
import sys
import numpy as np

# sys.path.extend(["X:\\Staff\\shr015\\Software\\Optimization_SCE\\src\\sce"])
sys.path.append(os.path.abspath(".\\src\\sce"))
from sceua import sce_optim  # pylint: disable=wrong-import-position
from SCE_functioncall import *  # pylint: disable=wrong-import-position


PRECISION = 3
################################################################################
# PARAMETERS TO TUNE THE ALGORITHM
# Definition:
#  iseed = the random seed number (for repetetive testing purpose;pos integers)
#  iniflg = flag for initial parameter array (=1, included it in initial
#           population; otherwise, not included)
#  ngs = number of complexes (sub-populations)
# peps = value of NORMALIZED GEOMETRIC RANGE needed for convergence
#  maxn = maximum number of function evaluations allowed during optimization
#  kstop = maximum number of evolution loops before convergency
#  pcento = the percentage change allowed in kstop loops before convergency
# -------------------------------------------------------------------------------
maxn = 10000
kstop = 30
pcento = 0.001
peps = 0.001
iseed = 0
iniflg = 0
ngs = 5
# bl=np.array([-2,-2])
# bu=np.array([2,2])
# x0=np.array([2,2])
# funct = testfunctn1
# bestx,bestf,BESTX,BESTF,ICALL = sce_optim(funct,x0,args=(1,),bl=bl,bu=bu,maxn=maxn,kstop=kstop,pcento=pcento,peps=peps,ngs=ngs,iseed=iseed,iniflg=iniflg)
# expected_bestx = np.array([0.0,-1.0])
# expected_bestf = np.array(3.0)
# np.testing.assert_allclose(bestx, expected_bestx,rtol=1e-6, atol=1e-6)
# assert bestf.round(PRECISION) == expected_bestf.round(PRECISION)


def test_Goldstein_Price_funct():
    """
    Test the Goldstein-Price Function
    Bound X1=[-2,2], X2=[-2,2]; Global Optimum: 3.0,(0.0,-1.0)
    """
    bl = np.array([-2, -2])
    bu = np.array([2, 2])
    x0 = np.array([2, 2])
    funct = Goldstein_Price_funct  # type: ignore
    bestx, bestf, _, _, _ = sce_optim(
        funct,
        x0,
        args=(1,),
        bl=bl,
        bu=bu,
        maxn=maxn,
        kstop=kstop,
        pcento=pcento,
        peps=peps,
        ngs=ngs,
        iseed=iseed,
        iniflg=iniflg,
    )

    expected_bestx = np.array([0.0, -1.0])
    expected_bestf = np.array(3.0)

    np.testing.assert_allclose(bestx, expected_bestx, rtol=1e-6, atol=1e-6)
    assert bestf.round(PRECISION) == expected_bestf.round(PRECISION)


def test_Rosenbrock_funct():
    """
    Test the Rosenbrock Function
    Bound: X1=[-5,5], X2=[-2,8]; Global Optimum: 0,(1,1)
        bl=[-5 -5]; bu=[5 5]; x0=[1 1];
    """
    bl = np.array([-5.0, -2.0])
    bu = np.array([5.0, 8.0])
    x0 = np.array([-2.0, 7.0])
    funct = Rosenbrock_funct  # type: ignore
    bestx, bestf, _, _, _ = sce_optim(
        funct,
        x0,
        args=(1,),
        bl=bl,
        bu=bu,
        maxn=maxn,
        kstop=kstop,
        pcento=pcento,
        peps=peps,
        ngs=ngs,
        iseed=iseed,
        iniflg=iniflg,
    )

    expected_bestx = np.array([1.0, 1.0])
    expected_bestf = np.array(0.0)

    np.testing.assert_allclose(bestx, expected_bestx, rtol=1e-6, atol=1e-6)
    assert bestf.round(PRECISION) == expected_bestf.round(PRECISION)


def test_Sixhump_camelback_funct():
    """
    Test the Six-hump Camelback Function.
    Bound: X1=[-5,5], X2=[-5,5]
    True Optima: -1.031628453489877, (-0.08983,0.7126), (0.08983,-0.7126)
    """
    bl = np.array([-2.0, -2.0])
    bu = np.array([2.0, 2.0])
    x0 = np.array([-1.0, 1.0])
    funct = Sixhump_camelback_funct  # type: ignore
    bestx, bestf, _, _, _ = sce_optim(
        funct,
        x0,
        args=(1,),
        bl=bl,
        bu=bu,
        maxn=maxn,
        kstop=kstop,
        pcento=pcento,
        peps=peps,
        ngs=ngs,
        iseed=iseed,
        iniflg=iniflg,
    )

    expected_bestx1 = np.array([-0.08983, 0.7126])
    expected_bestx2 = np.array([0.08983, -0.7126])
    expected_bestf = np.array(-1.031628453489877)

    assert bestf.round(PRECISION) == expected_bestf.round(PRECISION)
    tf1 = np.allclose(bestx, expected_bestx1, rtol=1e-4, atol=1e-4)
    tf2 = np.allclose(bestx, expected_bestx2, rtol=1e-4, atol=1e-4)
    assert tf1 or tf2


def test_Rastrigin_funct():
    """
    Test the Rastrigin Function.
    Bound: X1=[-1,1], X2=[-1,1]
    Global Optimum: -2, (0,0)
    """
    bl = np.array([-5.0, -5.0])
    bu = np.array([5.0, 5.0])
    x0 = np.array([-1.0, 1.0])
    funct = Rastrigin_funct  # type: ignore
    bestx, bestf, _, _, _ = sce_optim(
        funct,
        x0,
        args=(1,),
        bl=bl,
        bu=bu,
        maxn=maxn,
        kstop=kstop,
        pcento=pcento,
        peps=peps,
        ngs=ngs,
        iseed=iseed,
        iniflg=iniflg,
    )

    expected_bestx = np.array([0, 0])
    expected_bestf = np.array(-2.0)

    assert bestf.round(PRECISION) == expected_bestf.round(PRECISION)
    np.testing.assert_allclose(bestx, expected_bestx, rtol=1e-4, atol=1e-4)


def test_Griewank_funct():
    """
    Test the the Griewank Function (2-D or 10-D).
    Bound: X(i)=[-600,600], for i=1,2,...,10  !for visualization only 2!
    Global Optimum: 0, at origin
    """
    bl = -600 * np.ones(2)
    bu = 600 * np.ones(2)
    x0 = np.zeros(2)
    funct = Griewank_funct  # type: ignore
    bestx, bestf, _, _, _ = sce_optim(
        funct,
        x0,
        args=(2,),
        bl=bl,
        bu=bu,
        maxn=maxn,
        kstop=kstop,
        pcento=pcento,
        peps=peps,
        ngs=ngs,
        iseed=iseed,
        iniflg=iniflg,
    )

    expected_bestx = np.array([0.0, 0.0])
    expected_bestf = np.array(0.0)

    assert bestf.round(PRECISION) == expected_bestf.round(PRECISION)
    np.testing.assert_allclose(bestx, expected_bestx, rtol=1e-4, atol=1e-4)

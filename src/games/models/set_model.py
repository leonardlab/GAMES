#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 14:57:05 2022

@author: kate
"""
import json
from games.models.synTF import synTF
from games.models.synTF_chem import synTF_chem


def set_model():
    """
    Defines the model class to use depending on the modelID defined in Settings

    Parameters
    ----------
    None

    Returns
    -------
    model
        object defining the model

    """

    if settings["modelID"] == "synTF_chem":
        given_model = synTF_chem(
            parameters=settings["parameters"], mechanismID=settings["mechanismID"]
        )

    elif settings["modelID"] == "synTF":
        given_model = synTF(parameters=settings["parameters"])

    return given_model


file = open("/Users/kdreyer/Documents/Github/GAMES/src/games/config/config.json", encoding="utf-8")
settings = json.load(file)
model = set_model()

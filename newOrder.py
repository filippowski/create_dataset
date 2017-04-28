#!/usr/bin/env python2
## -*- coding: utf-8 -*-

import numpy as np


def get_inverse_array(numpy_array):
    return np.fliplr([numpy_array])[0]

def get_new_order_features():
    # original segments
    l_eye_top = np.arange(0, 5)
    l_eye_bot = np.arange(5, 8)
    r_eye_top = np.arange(8, 13)
    r_eye_bot = np.arange(13, 16)
    l_cheek = np.arange(16, 19)
    r_cheek = np.arange(19, 22)
    chin = np.arange(22, 25)
    chin_middle = np.arange(25, 28)
    # mirrors of segments
    l_eye_top_mirror = get_inverse_array(r_eye_top)
    l_eye_bot_mirror = get_inverse_array(r_eye_bot)
    r_eye_top_mirror = get_inverse_array(l_eye_top)
    r_eye_bot_mirror = get_inverse_array(l_eye_bot)
    l_cheek_mirror = r_cheek
    r_cheek_mirror = l_cheek
    chin_mirror = get_inverse_array(chin)
    chin_middle_mirror = chin_middle

    new_order_features = np.hstack((l_eye_top_mirror, l_eye_bot_mirror, r_eye_top_mirror, r_eye_bot_mirror,
                                           l_cheek_mirror, r_cheek_mirror, chin_mirror, chin_middle_mirror))
    return new_order_features

def get_new_order_28():
    new_order_28 = [1, 0, 8, 13, 12, 11, 10, 9, 2, 7, 6, 5, 4, 3, 27, 26, 25, 24, 23, 22, 20, 21, 19, 18, 17, 16, 15,
                    14]
    return new_order_28

def get_new_order_85():
    # defs of face fragments (inversed order)
    face_oval = get_inverse_array(np.arange(0, 17))
    forehead = get_inverse_array(np.arange(17, 20))
    l_eyebrow = np.arange(20, 28)
    l_eye = np.arange(28, 34)
    nose_bridge = get_inverse_array(np.arange(34, 37))
    r_eyebrow = np.arange(37, 45)
    r_eye = np.arange(45, 51)
    nose = get_inverse_array(np.arange(51, 65))
    mouth = recursive_append(
        [np.arange(71, 64, -1), np.arange(76, 71, -1), np.arange(81, 76, -1), np.arange(84, 81, -1)])

    new_order_85 = np.hstack((face_oval, forehead, r_eyebrow, r_eye, nose_bridge, l_eyebrow, l_eye, nose, mouth))
    # changing nose central point
    tmp = new_order_85[57]
    new_order_85[57] = new_order_85[58]
    new_order_85[58] = tmp

    return new_order_85


def get_new_order_119():
    # new order 119 points
    # original segments
    # basic face landmarks points
    face_oval = np.arange(0, 17)
    forehead = np.arange(17, 20)
    l_eyebrow = np.arange(20, 28)
    l_eye = np.arange(28, 36)
    nose_bridge = np.arange(36, 39)
    r_eyebrow = np.arange(39, 47)
    r_eye = np.arange(47, 55)
    nose = np.arange(55, 69)
    mouth = np.arange(69, 89)
    # face features points
    l_eye_top = np.arange(89, 94)
    l_eye_bot = np.arange(94, 97)
    r_eye_top = np.arange(97, 102)
    r_eye_bot = np.arange(102, 105)
    l_cheek = np.arange(105, 108)
    l_cheek_comma = np.array([108])
    r_cheek = np.arange(109, 112)
    r_cheek_comma = np.array([112])
    chin = np.arange(113, 116)
    chin_middle = np.arange(116, 119)

    # mirrors of segments
    face_oval_mirror = get_inverse_array(face_oval)
    forehead_mirror = get_inverse_array(forehead)
    r_eyebrow_mirror = r_eyebrow
    r_eye_mirror = r_eye
    nose_bridge_mirror = get_inverse_array(nose_bridge)
    l_eyebrow_mirror = l_eyebrow
    l_eye_mirror = l_eye
    nose_mirror = get_inverse_array(nose)
    mouth_mirror = np.hstack((np.arange(75, 68, -1), np.arange(80, 75, -1), np.arange(85, 80, -1), np.arange(88, 85, -1)))
    # face features points
    l_eye_top_mirror = get_inverse_array(r_eye_top)
    l_eye_bot_mirror = get_inverse_array(r_eye_bot)
    r_eye_top_mirror = get_inverse_array(l_eye_top)
    r_eye_bot_mirror = get_inverse_array(l_eye_bot)
    l_cheek_mirror = r_cheek
    l_cheek_comma_mirror = r_cheek_comma
    r_cheek_mirror = l_cheek
    r_cheek_comma_mirror = l_cheek_comma
    chin_mirror = get_inverse_array(chin)
    chin_middle_mirror = chin_middle

    # new_order 119 points
    new_order_119 = np.hstack((
         face_oval_mirror, forehead_mirror, r_eyebrow_mirror, r_eye_mirror, nose_bridge_mirror,
         l_eyebrow_mirror, l_eye_mirror, nose_mirror, mouth_mirror, l_eye_top_mirror,
         l_eye_bot_mirror, r_eye_top_mirror, r_eye_bot_mirror,
         l_cheek_mirror, l_cheek_comma_mirror, r_cheek_mirror, r_cheek_comma_mirror,
         chin_mirror, chin_middle_mirror))
    # changing nose central point
    tmp = new_order_119[61]
    new_order_119[61] = new_order_119[62]
    new_order_119[62] = tmp
    return new_order_119
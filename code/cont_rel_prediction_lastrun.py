#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.2),
    on Wed Feb 26 16:49:42 2025
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.2'
expName = 'cont_rel_prediction'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'../data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/hpopal/Library/CloudStorage/GoogleDrive-hpopal@umd.edu/My Drive/dscn_lab/projects/ocean/derivatives/task-cont_rel_pred/code/cont_rel_prediction_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1920, 1080], fullscr=True, screen=0,
            winType='pyglet', allowStencil=True,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "intro_main" ---
    intro_main_prompt = visual.TextStim(win=win, name='intro_main_prompt',
        text='Hello. You will be completing a task that will ask you to watch video clips and to respond with the relationship you believe the characters in the clip share.\n\nPress the space bar to continue.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    intro_main_cont = keyboard.Keyboard()
    
    # --- Initialize components for Routine "intro_video" ---
    intro_video_prompt = visual.TextStim(win=win, name='intro_video_prompt',
        text='While watching a video clip, you can make a response at any time and then submit that response. You can do this throughout the entire clip. Please submit a response when you first have a guess as to what the relationship is, and again, as much as necessary, when you wish to change your answer.\n\nPress the space bar to continue.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    intro_video_cont = keyboard.Keyboard()
    
    # --- Initialize components for Routine "intro_resp" ---
    intro_resp_prompt = visual.TextStim(win=win, name='intro_resp_prompt',
        text='After each clip, you will be asked to make one final guess and hit submit. Even if your answer has not changed, please type your response and hit submit. \n\nPress the space bar to start the task.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    intro_resp_cont = keyboard.Keyboard()
    
    # --- Initialize components for Routine "video_phase" ---
    video = visual.MovieStim(
        win, name='video',
        filename=None, movieLib='ffpyplayer',
        loop=False, volume=1.0, noAudio=False,
        pos=(0, 0), size=(0.5, 0.5), units=win.units,
        ori=0.0, anchor='center',opacity=None, contrast=1.0,
        depth=0
    )
    video_resp = visual.TextBox2(
         win, text=None, placeholder='', font='Arial',
         pos=(0, -0.4),     letterHeight=0.05,
         size=(0.5, 0.1), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor='white', borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='video_resp',
         depth=-1, autoLog=True,
    )
    button_video_resp = visual.ButtonStim(win, 
        text='', font='Arvo',
        pos=(0.4, -0.4),
        letterHeight=0.05,
        size=(0.3, 0.1), borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button_video_resp',
        depth=-2
    )
    button_video_resp.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "resp_phase" ---
    resp_prompt = visual.TextStim(win=win, name='resp_prompt',
        text='What relationship did the characters share?',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    post_video_resp = visual.TextBox2(
         win, text=None, placeholder='Type here...', font='Arial',
         pos=(0, -0.4),     letterHeight=0.05,
         size=(0.5, 0.1), borderWidth=2.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor='white', borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='post_video_resp',
         depth=-1, autoLog=True,
    )
    button_post_video = visual.ButtonStim(win, 
        text='Submit', font='Arvo',
        pos=(0.4, -0.4),
        letterHeight=0.05,
        size=(0.3, 0.1), borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button_post_video',
        depth=-2
    )
    button_post_video.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "fixation" ---
    fixation_stim = visual.TextStim(win=win, name='fixation_stim',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "end_task" ---
    text = visual.TextStim(win=win, name='text',
        text='Thank you for completing the task. Press the space bar to end the task.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard()
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "intro_main" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('intro_main.started', globalClock.getTime())
    intro_main_cont.keys = []
    intro_main_cont.rt = []
    _intro_main_cont_allKeys = []
    # keep track of which components have finished
    intro_mainComponents = [intro_main_prompt, intro_main_cont]
    for thisComponent in intro_mainComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "intro_main" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *intro_main_prompt* updates
        
        # if intro_main_prompt is starting this frame...
        if intro_main_prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_main_prompt.frameNStart = frameN  # exact frame index
            intro_main_prompt.tStart = t  # local t and not account for scr refresh
            intro_main_prompt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_main_prompt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_main_prompt.started')
            # update status
            intro_main_prompt.status = STARTED
            intro_main_prompt.setAutoDraw(True)
        
        # if intro_main_prompt is active this frame...
        if intro_main_prompt.status == STARTED:
            # update params
            pass
        
        # *intro_main_cont* updates
        waitOnFlip = False
        
        # if intro_main_cont is starting this frame...
        if intro_main_cont.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_main_cont.frameNStart = frameN  # exact frame index
            intro_main_cont.tStart = t  # local t and not account for scr refresh
            intro_main_cont.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_main_cont, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_main_cont.started')
            # update status
            intro_main_cont.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(intro_main_cont.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(intro_main_cont.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if intro_main_cont.status == STARTED and not waitOnFlip:
            theseKeys = intro_main_cont.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _intro_main_cont_allKeys.extend(theseKeys)
            if len(_intro_main_cont_allKeys):
                intro_main_cont.keys = _intro_main_cont_allKeys[-1].name  # just the last key pressed
                intro_main_cont.rt = _intro_main_cont_allKeys[-1].rt
                intro_main_cont.duration = _intro_main_cont_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro_mainComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro_main" ---
    for thisComponent in intro_mainComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('intro_main.stopped', globalClock.getTime())
    # check responses
    if intro_main_cont.keys in ['', [], None]:  # No response was made
        intro_main_cont.keys = None
    thisExp.addData('intro_main_cont.keys',intro_main_cont.keys)
    if intro_main_cont.keys != None:  # we had a response
        thisExp.addData('intro_main_cont.rt', intro_main_cont.rt)
        thisExp.addData('intro_main_cont.duration', intro_main_cont.duration)
    thisExp.nextEntry()
    # the Routine "intro_main" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "intro_video" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('intro_video.started', globalClock.getTime())
    intro_video_cont.keys = []
    intro_video_cont.rt = []
    _intro_video_cont_allKeys = []
    # keep track of which components have finished
    intro_videoComponents = [intro_video_prompt, intro_video_cont]
    for thisComponent in intro_videoComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "intro_video" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *intro_video_prompt* updates
        
        # if intro_video_prompt is starting this frame...
        if intro_video_prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_video_prompt.frameNStart = frameN  # exact frame index
            intro_video_prompt.tStart = t  # local t and not account for scr refresh
            intro_video_prompt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_video_prompt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_video_prompt.started')
            # update status
            intro_video_prompt.status = STARTED
            intro_video_prompt.setAutoDraw(True)
        
        # if intro_video_prompt is active this frame...
        if intro_video_prompt.status == STARTED:
            # update params
            pass
        
        # *intro_video_cont* updates
        waitOnFlip = False
        
        # if intro_video_cont is starting this frame...
        if intro_video_cont.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_video_cont.frameNStart = frameN  # exact frame index
            intro_video_cont.tStart = t  # local t and not account for scr refresh
            intro_video_cont.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_video_cont, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_video_cont.started')
            # update status
            intro_video_cont.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(intro_video_cont.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(intro_video_cont.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if intro_video_cont.status == STARTED and not waitOnFlip:
            theseKeys = intro_video_cont.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _intro_video_cont_allKeys.extend(theseKeys)
            if len(_intro_video_cont_allKeys):
                intro_video_cont.keys = _intro_video_cont_allKeys[-1].name  # just the last key pressed
                intro_video_cont.rt = _intro_video_cont_allKeys[-1].rt
                intro_video_cont.duration = _intro_video_cont_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro_videoComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro_video" ---
    for thisComponent in intro_videoComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('intro_video.stopped', globalClock.getTime())
    # check responses
    if intro_video_cont.keys in ['', [], None]:  # No response was made
        intro_video_cont.keys = None
    thisExp.addData('intro_video_cont.keys',intro_video_cont.keys)
    if intro_video_cont.keys != None:  # we had a response
        thisExp.addData('intro_video_cont.rt', intro_video_cont.rt)
        thisExp.addData('intro_video_cont.duration', intro_video_cont.duration)
    thisExp.nextEntry()
    # the Routine "intro_video" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "intro_resp" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('intro_resp.started', globalClock.getTime())
    intro_resp_cont.keys = []
    intro_resp_cont.rt = []
    _intro_resp_cont_allKeys = []
    # keep track of which components have finished
    intro_respComponents = [intro_resp_prompt, intro_resp_cont]
    for thisComponent in intro_respComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "intro_resp" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *intro_resp_prompt* updates
        
        # if intro_resp_prompt is starting this frame...
        if intro_resp_prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_resp_prompt.frameNStart = frameN  # exact frame index
            intro_resp_prompt.tStart = t  # local t and not account for scr refresh
            intro_resp_prompt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_resp_prompt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_resp_prompt.started')
            # update status
            intro_resp_prompt.status = STARTED
            intro_resp_prompt.setAutoDraw(True)
        
        # if intro_resp_prompt is active this frame...
        if intro_resp_prompt.status == STARTED:
            # update params
            pass
        
        # *intro_resp_cont* updates
        waitOnFlip = False
        
        # if intro_resp_cont is starting this frame...
        if intro_resp_cont.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_resp_cont.frameNStart = frameN  # exact frame index
            intro_resp_cont.tStart = t  # local t and not account for scr refresh
            intro_resp_cont.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_resp_cont, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_resp_cont.started')
            # update status
            intro_resp_cont.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(intro_resp_cont.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(intro_resp_cont.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if intro_resp_cont.status == STARTED and not waitOnFlip:
            theseKeys = intro_resp_cont.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _intro_resp_cont_allKeys.extend(theseKeys)
            if len(_intro_resp_cont_allKeys):
                intro_resp_cont.keys = _intro_resp_cont_allKeys[-1].name  # just the last key pressed
                intro_resp_cont.rt = _intro_resp_cont_allKeys[-1].rt
                intro_resp_cont.duration = _intro_resp_cont_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro_respComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro_resp" ---
    for thisComponent in intro_respComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('intro_resp.stopped', globalClock.getTime())
    # check responses
    if intro_resp_cont.keys in ['', [], None]:  # No response was made
        intro_resp_cont.keys = None
    thisExp.addData('intro_resp_cont.keys',intro_resp_cont.keys)
    if intro_resp_cont.keys != None:  # we had a response
        thisExp.addData('intro_resp_cont.rt', intro_resp_cont.rt)
        thisExp.addData('intro_resp_cont.duration', intro_resp_cont.duration)
    thisExp.nextEntry()
    # the Routine "intro_resp" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('../stimuli/stimuli_list.csv'),
        seed=None, name='trials')
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "video_phase" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('video_phase.started', globalClock.getTime())
        video.setMovie(stimuli)
        video_resp.reset()
        video_resp.setText('')
        video_resp.setPlaceholder('Type here...')
        button_video_resp.setText('Submit')
        # reset button_video_resp to account for continued clicks & clear times on/off
        button_video_resp.reset()
        # Run 'Begin Routine' code from code
        video_resp.text_full = []
        # keep track of which components have finished
        video_phaseComponents = [video, video_resp, button_video_resp]
        for thisComponent in video_phaseComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "video_phase" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *video* updates
            
            # if video is starting this frame...
            if video.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                video.frameNStart = frameN  # exact frame index
                video.tStart = t  # local t and not account for scr refresh
                video.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(video, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'video.started')
                # update status
                video.status = STARTED
                video.setAutoDraw(True)
                video.play()
            if video.isFinished:  # force-end the Routine
                continueRoutine = False
            
            # *video_resp* updates
            
            # if video_resp is starting this frame...
            if video_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                video_resp.frameNStart = frameN  # exact frame index
                video_resp.tStart = t  # local t and not account for scr refresh
                video_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(video_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'video_resp.started')
                # update status
                video_resp.status = STARTED
                video_resp.setAutoDraw(True)
            
            # if video_resp is active this frame...
            if video_resp.status == STARTED:
                # update params
                pass
            # *button_video_resp* updates
            
            # if button_video_resp is starting this frame...
            if button_video_resp.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                button_video_resp.frameNStart = frameN  # exact frame index
                button_video_resp.tStart = t  # local t and not account for scr refresh
                button_video_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(button_video_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'button_video_resp.started')
                # update status
                button_video_resp.status = STARTED
                button_video_resp.setAutoDraw(True)
            
            # if button_video_resp is active this frame...
            if button_video_resp.status == STARTED:
                # update params
                pass
                # check whether button_video_resp has been pressed
                if button_video_resp.isClicked:
                    if not button_video_resp.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        button_video_resp.timesOn.append(button_video_resp.buttonClock.getTime())
                        button_video_resp.timesOff.append(button_video_resp.buttonClock.getTime())
                    elif len(button_video_resp.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        button_video_resp.timesOff[-1] = button_video_resp.buttonClock.getTime()
                    if not button_video_resp.wasClicked:
                        # run callback code when button_video_resp is clicked
                        video_resp.text_full.append(video_resp.text)
                        video_resp.text = ''
            # take note of whether button_video_resp was clicked, so that next frame we know if clicks are new
            button_video_resp.wasClicked = button_video_resp.isClicked and button_video_resp.status == STARTED
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in video_phaseComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "video_phase" ---
        for thisComponent in video_phaseComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('video_phase.stopped', globalClock.getTime())
        video.stop()  # ensure movie has stopped at end of Routine
        trials.addData('video_resp.text',video_resp.text)
        trials.addData('button_video_resp.numClicks', button_video_resp.numClicks)
        if button_video_resp.numClicks:
           trials.addData('button_video_resp.timesOn', button_video_resp.timesOn)
           trials.addData('button_video_resp.timesOff', button_video_resp.timesOff)
        else:
           trials.addData('button_video_resp.timesOn', "")
           trials.addData('button_video_resp.timesOff', "")
        # Run 'End Routine' code from code
        trials.addData('video_resp.text_full',video_resp.text_full)
        # the Routine "video_phase" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "resp_phase" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('resp_phase.started', globalClock.getTime())
        post_video_resp.reset()
        # reset button_post_video to account for continued clicks & clear times on/off
        button_post_video.reset()
        # keep track of which components have finished
        resp_phaseComponents = [resp_prompt, post_video_resp, button_post_video]
        for thisComponent in resp_phaseComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "resp_phase" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *resp_prompt* updates
            
            # if resp_prompt is starting this frame...
            if resp_prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                resp_prompt.frameNStart = frameN  # exact frame index
                resp_prompt.tStart = t  # local t and not account for scr refresh
                resp_prompt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(resp_prompt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'resp_prompt.started')
                # update status
                resp_prompt.status = STARTED
                resp_prompt.setAutoDraw(True)
            
            # if resp_prompt is active this frame...
            if resp_prompt.status == STARTED:
                # update params
                pass
            
            # *post_video_resp* updates
            
            # if post_video_resp is starting this frame...
            if post_video_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                post_video_resp.frameNStart = frameN  # exact frame index
                post_video_resp.tStart = t  # local t and not account for scr refresh
                post_video_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(post_video_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'post_video_resp.started')
                # update status
                post_video_resp.status = STARTED
                post_video_resp.setAutoDraw(True)
            
            # if post_video_resp is active this frame...
            if post_video_resp.status == STARTED:
                # update params
                pass
            # *button_post_video* updates
            
            # if button_post_video is starting this frame...
            if button_post_video.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                button_post_video.frameNStart = frameN  # exact frame index
                button_post_video.tStart = t  # local t and not account for scr refresh
                button_post_video.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(button_post_video, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'button_post_video.started')
                # update status
                button_post_video.status = STARTED
                button_post_video.setAutoDraw(True)
            
            # if button_post_video is active this frame...
            if button_post_video.status == STARTED:
                # update params
                pass
                # check whether button_post_video has been pressed
                if button_post_video.isClicked:
                    if not button_post_video.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        button_post_video.timesOn.append(button_post_video.buttonClock.getTime())
                        button_post_video.timesOff.append(button_post_video.buttonClock.getTime())
                    elif len(button_post_video.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        button_post_video.timesOff[-1] = button_post_video.buttonClock.getTime()
                    if not button_post_video.wasClicked:
                        # end routine when button_post_video is clicked
                        continueRoutine = False
                    if not button_post_video.wasClicked:
                        # run callback code when button_post_video is clicked
                        pass
            # take note of whether button_post_video was clicked, so that next frame we know if clicks are new
            button_post_video.wasClicked = button_post_video.isClicked and button_post_video.status == STARTED
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in resp_phaseComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "resp_phase" ---
        for thisComponent in resp_phaseComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('resp_phase.stopped', globalClock.getTime())
        trials.addData('post_video_resp.text',post_video_resp.text)
        trials.addData('button_post_video.numClicks', button_post_video.numClicks)
        if button_post_video.numClicks:
           trials.addData('button_post_video.timesOn', button_post_video.timesOn)
           trials.addData('button_post_video.timesOff', button_post_video.timesOff)
        else:
           trials.addData('button_post_video.timesOn', "")
           trials.addData('button_post_video.timesOff', "")
        # the Routine "resp_phase" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "fixation" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('fixation.started', globalClock.getTime())
        # keep track of which components have finished
        fixationComponents = [fixation_stim]
        for thisComponent in fixationComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation_stim* updates
            
            # if fixation_stim is starting this frame...
            if fixation_stim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_stim.frameNStart = frameN  # exact frame index
                fixation_stim.tStart = t  # local t and not account for scr refresh
                fixation_stim.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_stim, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_stim.started')
                # update status
                fixation_stim.status = STARTED
                fixation_stim.setAutoDraw(True)
            
            # if fixation_stim is active this frame...
            if fixation_stim.status == STARTED:
                # update params
                pass
            
            # if fixation_stim is stopping this frame...
            if fixation_stim.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation_stim.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation_stim.tStop = t  # not accounting for scr refresh
                    fixation_stim.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation_stim.stopped')
                    # update status
                    fixation_stim.status = FINISHED
                    fixation_stim.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixationComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation" ---
        for thisComponent in fixationComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('fixation.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials'
    
    # get names of stimulus parameters
    if trials.trialList in ([], [None], None):
        params = []
    else:
        params = trials.trialList[0].keys()
    # save data for this loop
    trials.saveAsText(filename + 'trials.csv', delim=',',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "end_task" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('end_task.started', globalClock.getTime())
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # keep track of which components have finished
    end_taskComponents = [text, key_resp]
    for thisComponent in end_taskComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end_task" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end_taskComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end_task" ---
    for thisComponent in end_taskComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('end_task.stopped', globalClock.getTime())
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "end_task" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)

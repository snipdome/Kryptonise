# 
# This file is part of the Kryptonise distribution (https://github.com/snipdome/Kryptonise).
# Copyright (c) 2024 imec-Vision Lab, University of Antwerp.
# 
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.
 
import logging
import os
import copy
from typing import Annotated, Optional

import vtk
import vtkSegmentationCorePython as vtkSegmentationCore 
import vtkSlicerSegmentationsModuleLogicPython as vtkSlicerSegmentationsModuleLogic
import vtkITK

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
	parameterNodeWrapper,
	WithinRange,
)
from slicer import vtkMRMLScalarVolumeNode, vtkMRMLSegmentationNode, vtkMRMLModelNode


import numpy as np
import time
import threading
from enum import Enum

from skimage.segmentation import flood, flood_fill #FIXME remove these imports
from matplotlib import pyplot as plt #FIXME remove these imports

import SimpleITK as sitk
from vtk.util import numpy_support
import sitkUtils

#
# Kryptonise
#


class Kryptonise(ScriptedLoadableModule):

	def __init__(self, parent):
		ScriptedLoadableModule.__init__(self, parent)
		self.parent.title = _("Kryptonise")  
		self.parent.categories = [translate("qSlicerAbstractCoreModule", "Post-Process")]
		self.parent.dependencies = []
		self.parent.contributors = ["Domenico Iuso (imec-Visionlab, UAntwerp)"] 
		self.parent.helpText = _("""
This is a scripted loadable module for the mesh extraction and reduction. It also roto-translates the mesh in Krypton reference frame.
See more information in <a href="https://github.com/snipdome/Kryptonise">module documentation</a>.
""")
		self.parent.acknowledgementText = _("""
This file was originally developed by Domenico Iuso (imec-Visionlab, UAntwerp). 
""")

		# Additional initialization step after application startup is complete
		slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
	"""Add data sets to Sample Data module."""
	# It is always recommended to provide sample data for users to make it easy to try the module,
	# but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

	import SampleData

	iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

	# To ensure that the source code repository remains small (can be downloaded and installed quickly)
	# it is recommended to store data sets that are larger than a few MB in a Github release.

	# Kryptonise1
	SampleData.SampleDataLogic.registerCustomSampleDataSource(
		# Category and sample name displayed in Sample Data module
		category="Kryptonise",
		sampleName="Kryptonise1",
		# Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
		# It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
		thumbnailFileName=os.path.join(iconsPath, "Kryptonise1.png"),
		# Download URL and target file name
		uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
		fileNames="Kryptonise1.nrrd",
		# Checksum to ensure file integrity. Can be computed by this command:
		#  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
		checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
		# This node name will be used when the data set is loaded
		nodeNames="Kryptonise1",
	)

	# Kryptonise2
	SampleData.SampleDataLogic.registerCustomSampleDataSource(
		# Category and sample name displayed in Sample Data module
		category="Kryptonise",
		sampleName="Kryptonise2",
		thumbnailFileName=os.path.join(iconsPath, "Kryptonise2.png"),
		# Download URL and target file name
		uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
		fileNames="Kryptonise2.nrrd",
		checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
		# This node name will be used when the data set is loaded
		nodeNames="Kryptonise2",
	)


#
# KryptoniseParameterNode
#


@parameterNodeWrapper
class KryptoniseParameterNode:
	"""
	The parameters needed by module.

	inputVolume - The volume to threshold.
	outputLabelMap - The output label map, where the segmented volume will be written.
	"""

	inputLabelMap: vtkMRMLSegmentationNode
	outputMesh: vtkMRMLModelNode
	#imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
	


#
# KryptoniseWidget
#


class KryptoniseWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
	"""Uses ScriptedLoadableModuleWidget base class, available at:
	https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
	"""

	def __init__(self, parent=None) -> None:
		"""Called when the user opens the module the first time and the widget is initialized."""
		ScriptedLoadableModuleWidget.__init__(self, parent)
		VTKObservationMixin.__init__(self)  # needed for parameter node observation
		self.logic = None
		self._parameterNode = None
		self._parameterNodeGuiTag = None

	def setup(self) -> None:
		"""Called when the user opens the module the first time and the widget is initialized."""
		ScriptedLoadableModuleWidget.setup(self)

		# Load widget from .ui file (created by Qt Designer).
		# Additional widgets can be instantiated manually and added to self.layout.
		uiWidget = slicer.util.loadUI(self.resourcePath("UI/Kryptonise.ui"))
		self.layout.addWidget(uiWidget)
		self.ui = slicer.util.childWidgetVariables(uiWidget)

		# hide the debug level slider
		self.ui.debugLevelSliderWidget.hide()
		self.ui.debugLevelLabel.hide()

		# Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
		# "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
		# "setMRMLScene(vtkMRMLScene*)" slot.
		uiWidget.setMRMLScene(slicer.mrmlScene)

		# Create logic class. Logic implements all computations that should be possible to run
		# in batch mode, without a graphical user interface.
		self.logic = KryptoniseLogic()

		# Connections

		# These connections ensure that we update parameter node when scene is closed
		self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
		self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

		# Buttons
		self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

		# Make sure parameter node is initialized (needed for module reload)
		self.initializeParameterNode()

	def cleanup(self) -> None:
		"""Called when the application closes and the module widget is destroyed."""
		self.removeObservers()

	def enter(self) -> None:
		"""Called each time the user opens this module."""
		# Make sure parameter node exists and observed
		self.initializeParameterNode()

	def exit(self) -> None:
		"""Called each time the user opens a different module."""
		# Do not react to parameter node changes (GUI will be updated when the user enters into the module)
		if self._parameterNode:
			self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
			self._parameterNodeGuiTag = None
			self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

	def onSceneStartClose(self, caller, event) -> None:
		"""Called just before the scene is closed."""
		# Parameter node will be reset, do not use it anymore
		self.setParameterNode(None)

	def onSceneEndClose(self, caller, event) -> None:
		"""Called just after the scene is closed."""
		# If this module is shown while the scene is closed then recreate a new parameter node immediately
		if self.parent.isEntered:
			self.initializeParameterNode()

	def initializeParameterNode(self) -> None:
		"""Ensure parameter node exists and observed."""
		# Parameter node stores all user choices in parameter values, node selections, etc.
		# so that when the scene is saved and reloaded, these settings are restored.

		self.setParameterNode(self.logic.getParameterNode())

		# Select default input nodes if nothing is selected yet to save a few clicks for the user
		if not self._parameterNode.inputLabelMap:
			firstLabelMapNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
			if firstLabelMapNode:
				self._parameterNode.inputLabelMap = firstLabelMapNode
		
		self.try_update_output_mesh()
		
	def try_update_output_mesh(self):
		# Create a new mesh node to store the output if there is none yet with the same name as the input volume
		#if not self._parameterNode.outputLabelMap and self._parameterNode.inputVolume:
		if not self._parameterNode.outputMesh and self._parameterNode.inputLabelMap:
			input_name = self._parameterNode.inputLabelMap.GetName()
			meshNode = slicer.util.getFirstNodeByClassByName("vtkMRMLModelNode", input_name)
			if not meshNode:
				meshNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
				meshNode.SetName(input_name)
			self._parameterNode.outputMesh = meshNode


	def setParameterNode(self, inputParameterNode: Optional[KryptoniseParameterNode]) -> None:
		"""
		Set and observe parameter node.
		Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
		"""

		if self._parameterNode:
			self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
			self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
		self._parameterNode = inputParameterNode
		if self._parameterNode:
			# Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
			# ui element that needs connection.
			self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
			self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
			self._checkCanApply()

	def _checkCanApply(self, caller=None, event=None) -> None:
		self.try_update_output_mesh()
		if self._parameterNode and self._parameterNode.inputLabelMap and self._parameterNode.outputMesh:
			self.ui.applyButton.toolTip = _("Execute the mesh extraction, reduction and roto-translation")
			self.ui.applyButton.enabled = True
		else:
			self.ui.applyButton.toolTip = _("Select input label map and output mesh nodes")
			self.ui.applyButton.enabled = False

	def onApplyButton(self) -> None:

		# if self.ui.borderDilateSliderWidget.value <= self.ui.scaleSliderWidget.value:
		# 	self.ui.borderDilateSliderWidget.value = self.ui.scaleSliderWidget.value +1
		# 	logging.warning("Border dilate radius should be greater than the scale. Setting it to scale + 1")

		"""Run processing when user clicks "Apply" button."""
		with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
			kwargs = {
				"inputLabelMap": self._parameterNode.inputLabelMap,
				"outputMesh": self._parameterNode.outputMesh,

				"target_vertex_count": self.ui.targetVertexCountSliderWidget.value,
				"roll": self.ui.rollSliderWidget.value,
				"pitch": self.ui.pitchSliderWidget.value,
				"yaw": self.ui.yawSliderWidget.value,
				"transl_x": self.ui.xSliderWidget.value,
				"transl_y": self.ui.ySliderWidget.value,
				"transl_z": self.ui.zSliderWidget.value,

				# "scale": self.ui.scaleSliderWidget.value,
				# "want_curvature_flow": self.ui.curvatureFlowCheckBox.checked,
				# "curvature_niter": self.ui.iterationsCVSliderWidget.value,
				# "curvature_timestep": self.ui.stepSizeCVSliderWidget.value,

				# "top_stone_seed_perc": self.ui.topStoneHeightSliderWidget.value,
				# "bottom_stone_seed_perc": self.ui.bottomStoneHeightSliderWidget.value,
				# "border_dilate_radius": self.ui.borderDilateSliderWidget.value,
				# "border_erode_radius": self.ui.borderErosionSliderWidget.value,
				# "seed_radius": self.ui.seedRadiusSliderWidget.value,
				# "glue_h": self.ui.glueHeightSliderWidget.value,
				"debug_level": self.ui.debugLevelSliderWidget.value,
			}

			# Compute output
			self.logic.process(**kwargs)

#
# KryptoniseLogic
#


class KryptoniseLogic(ScriptedLoadableModuleLogic):
	"""This class should implement all the actual
	computation done by your module.  The interface
	should be such that other python code can import
	this class and make use of the functionality without
	requiring an instance of the Widget.
	Uses ScriptedLoadableModuleLogic base class, available at:
	https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
	"""

	def __init__(self) -> None:
		"""Called when the logic class is instantiated. Can be used for initializing member variables."""
		ScriptedLoadableModuleLogic.__init__(self)

	def getParameterNode(self):
		return KryptoniseParameterNode(super().getParameterNode())

	def process(self,
				inputLabelMap: vtkMRMLSegmentationNode,
				outputMesh: vtkMRMLModelNode,
				debug_level: int = 0,
				target_vertex_count: int = 1e5,
				roll: float = 0,
				pitch: float = 0,
				yaw: float = 0,
				transl_x: float = 0,
				transl_y: float = 0,
				transl_z: float = 0,
				) -> None:
		"""
		Run the processing algorithm.
		Can be used without GUI widget.
		:param 
		:param modify_input: if True then input volume will be modified, if False nothing will be changed
		
		"""

		if not inputLabelMap or not outputMesh:
			raise ValueError("Input or output volume is invalid")

		import time

		startTime = time.time()
		logging.info("Processing started")

		# Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
		cliParams = {
			"inputLabelMap": inputLabelMap.GetID(),
			"outputMesh": outputMesh.GetID(),
			"debug_level": int(debug_level),
			"target_vertex_count": int(target_vertex_count),
			"roll": float(roll*np.pi/180),
			"pitch": float(pitch*np.pi/180),
			"yaw": float(yaw*np.pi/180),
			"transl_x": float(transl_x),
			"transl_y": float(transl_y),
			"transl_z": float(transl_z),
		}
		#print(f"cliParams: {cliParams}")

		# Dome
		# cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
		# # We don't need the CLI module node anymore, remove it to not clutter the scene with it
		# slicer.mrmlScene.RemoveNode(cliNode)
		process_kernel(**cliParams)


		stopTime = time.time()
		logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")


#
# KryptoniseTest
#


class KryptoniseTest(ScriptedLoadableModuleTest):
	"""
	This is the test case for your scripted module.
	Uses ScriptedLoadableModuleTest base class, available at:
	https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
	"""

	def setUp(self):
		"""Do whatever is needed to reset the state - typically a scene clear will be enough."""
		slicer.mrmlScene.Clear()

	def runTest(self):
		"""Run as few or as many tests as needed here."""
		self.setUp()
		self.test_Kryptonise1()

	def test_Kryptonise1(self):
		"""Ideally you should have several levels of tests.  At the lowest level
		tests should exercise the functionality of the logic with different inputs
		(both valid and invalid).  At higher levels your tests should emulate the
		way the user would interact with your code and confirm that it still works
		the way you intended.
		One of the most important features of the tests is that it should alert other
		developers when their changes will have an impact on the behavior of your
		module.  For example, if a developer removes a feature that you depend on,
		your test should break so they know that the feature is needed.
		"""
		self.delayDisplay("Starting the test")
		import os
		if os.name == 'nt':
			dataset_path = ''
			raise ValueError("Fill in the dataset path")
		else:
			dataset_path = ''
			raise ValueError("Fill in the dataset path")
		masterVolumeNode = slicer.util.loadVolume(dataset_path + '.tif')
		outputLabelMap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")

		self.delayDisplay("Loaded test data set")

		kwargs = {
			"inputVolume": masterVolumeNode,
			"outputLabelMap": outputLabelMap,
			"scale": 4,
			"want_curvature_flow": False,
			"debug_level": 0,
		}
		# Test the module logic
		logic = KryptoniseLogic()
		logic.process(**kwargs)
		

## debug: 0 no verbose // 1 just text verbose // 2 show intermediate segmentations // 3: save images
class DebugLevel(Enum):
	NO_VERBOSE = 0
	TEXT_VERBOSE = 1
	SHOW_INTERM_SEGMS = 2
	SAVE_IMAGES = 3

colours = {
	"Background": [0.0,0.0,1.0],
	"Glue": [1.0,0.6,0.],
	"Stone": [1.0,1.0,0.3],
	"Border": [1.0,0.0,0.]
}


def checkInputs(**kwargs):
	sanitised_kwargs = kwargs.copy()

	return sanitised_kwargs

# def MeasureStats(node):
# 	import SegmentStatistics
# 	segStatLogic = SegmentStatistics.SegmentStatisticsLogic()
# 	segStatLogic.getParameterNode().SetParameter("Segmentation", node.GetID())
# 	segStatLogic.computeStatistics()
# 	stats = segStatLogic.getStatistics()
# 	# Display volume and surface area of each segment
# 	for segmentId in stats["SegmentIDs"]:
# 		volume_mm3 = stats[segmentId,"LabelmapSegmentStatisticsPlugin.volume_mm3"]
# 		segmentName = node.GetSegmentation().GetSegment(segmentId).GetName()
# 		#print(f"{segmentName} volume = {volume_mm3} mm3")
# 		logging.info(f"This stone has {volume_mm3*0.01757625:.2f} Ct\n\n")
		
def decimate_mesh(mesh, debug, **kwargs):

	decimate = vtk.vtkDecimatePro()
	decimate.SetInputData(mesh)
	decimate.SetTargetReduction(kwargs['target_reduction']) if 'target_reduction' in kwargs else None
	decimate.PreserveTopologyOn()
	logging.info(f"Decimating mesh with target reduction {kwargs['target_reduction']}. Starting with {mesh.GetNumberOfPoints()} points") if debug.value > DebugLevel.NO_VERBOSE.value else None
	decimate.Update()
	logging.info(f"Decimated mesh has {decimate.GetOutput().GetNumberOfPoints()} points") if debug.value > DebugLevel.NO_VERBOSE.value else None
	return decimate.GetOutput()

def select_biggest_connected_component(mesh, **kwargs):
	# get the biggest 
	connected = vtk.vtkPolyDataConnectivityFilter()
	connected.SetInputData(mesh)
	connected.SetExtractionModeToLargestRegion()
	connected.Update()
	return connected.GetOutput()

def get_data_directory( inputLabelMap, debug ):
		inputStorageNode = inputLabelMap.GetStorageNode()
		if inputStorageNode is None:
			logging.warning("No storage node found for the input labelmap. It will be inferred from the other nodes.")
			# open all the other nodes and get the paths of them. The longest one will be the output path
			inputStorageNodes = []
			for node in list(slicer.mrmlScene.GetNodesByClass("vtkMRMLStorageNode")) + list(slicer.mrmlScene.GetNodesByClass("vtkMRMLSegmentationNode")) + list(slicer.mrmlScene.GetNodesByClass("vtkMRMLScalarVolumeNode")):
				print(f'Checking node {node.GetID()} with name {node.GetName()}') if debug.value > 0 else None
				try:
					if node.GetStorageNode():
						inputStorageNodes.append(node.GetStorageNode())
				except:
					pass

			if len(inputStorageNodes) == 0:
				logging.warning("No storage node found in the scene. Defaulting to the desktop.")
				inputDirectory = str( os.path.join(os.path.expanduser('~'), 'Desktop') ) if os.name == 'nt' else str( os.path.expanduser('~') )
			else:
				inputDirectory = ""
				for node in inputStorageNodes:
					if len(node.GetFileName()) > len(inputDirectory):
						inputDirectory = os.path.dirname(node.GetFileName())
				logging.info(f"Input storage node found: {inputDirectory}") if debug.value > 0 else None
		else:
			inputDirectory = os.path.dirname(inputStorageNode.GetFileName())
		return inputDirectory

def get_info_from_log(logFile, debug):
	# This method reads the log file and extracts the roto-translation information
	#
	# Useful information: 
	# - Scanning position 
	# - First section
	# - Image Pixel Size (um)
	log_kwargs = {
		'scanning_pos': 'Scanning position',
		'proj_start': 'First Section',
		'vox_size': 'Scaled Image Pixel Size (um)',
		'num_of_files': 'Number Of Files',
		'pixel_size': 'Camera Pixel Size (um)',
		'n_width': 'Result Image Width (pixels)',
		'n_height': 'Result Image Height (pixels)',
	}
	out_kwargs = copy.deepcopy(log_kwargs)
	for line in logFile:
		for key, value in log_kwargs.items():
			# Check if the key is in the line, escaping spaces
			if line[0:len(value)] == value:
				#print(f"Found {value} in line {line}")
				new_value = line.split("=")[1].strip('\n')
				#print(f'New value: {new_value}') if debug.value > 0 else None
				while new_value.startswith(" "): # Strip the initial space, if present
					new_value = new_value[1:]
				#print(f'New value: {new_value}') if debug.value > 0 else None
				out_kwargs[key] = new_value.split(" ")[0]
				#print(f"Found {value} in line {line}. Extracted value: {out_kwargs[key]}") if debug.value > 0 else None

	logging.info(f"Log file information: {out_kwargs}") if debug.value > DebugLevel.NO_VERBOSE.value else None
	for key, value in out_kwargs.items():
		out_kwargs[key] = float(value)
	return out_kwargs

def get_scan_info(inputDirectory, debug):
		# Read the log file for information about the rototranslation. For this, get all the .log files in the directory inputDirectory
		
		logging.warning(" Input directory: " + inputDirectory)
		logFiles = []
		for file in os.listdir(inputDirectory):
			if file.endswith(".log"):
				logFiles.append(os.path.join(inputDirectory, file))
		if len(logFiles)==0:
			# Try to open the folder above
			new_inputDirectory = os.path.dirname(inputDirectory)
			for file in os.listdir(new_inputDirectory):
				if file.endswith(".log"):
					logFiles.append(os.path.join(new_inputDirectory, file))

		log_kwargs = None
		
		if len(logFiles) == 0: # open the first log file
			logging.warning("No log file found in the directory. The mesh will not be roto-translated.")
		else:
			logging.info(f"Opening log file {logFiles[0]}")
			with open(logFiles[0], "r") as logFile:
				log_kwargs = get_info_from_log(logFile, debug)
		return log_kwargs


def Rot_x(angle):
	# Rotation matrix around x
	return np.array([[1, 0, 0],
					 [0, np.cos(angle), -np.sin(angle)],
					 [0, np.sin(angle), np.cos(angle)]])


def Rot_y(angle):
	# Rotation matrix around y
	return np.array([[np.cos(angle), 0, np.sin(angle)],
					 [0, 1, 0],
					 [-np.sin(angle), 0, np.cos(angle)]])

def Rot_z(angle):
	# Rotation matrix around z
	return np.array([[np.cos(angle), -np.sin(angle), 0],
					 [np.sin(angle), np.cos(angle), 0],
					 [0, 0, 1]])

def CorrectMeshPosition(mesh, scanning_pos, proj_start, vox_size, pixel_size, num_of_files, n_width, n_height):

	roto_translation = np.eye(4, dtype=np.float32)

	magic_number = num_of_files # Probably from the number of slices in the total scan
	#voxel_scaling = pixel_size/vox_size
	h = 0.001 * vox_size * magic_number # vox_size is in um, the rest is in mm
	roto_translation[:3,3] += np.array([0, 0, -h], dtype=np.float32)

	shape = np.array([-n_width, n_height, 0], dtype=np.float32) # Possibly inverted because of the image coordinate system
	roto_translation[:3,3] += 0.001*vox_size * shape / 2 # vox_size is in um, the rest is in mm

	roto_translation[:3,3] += np.array([0, 0, scanning_pos], dtype=np.float32)

	roto_translation[:3,3] *= -1 # Invert the translation to have the mesh in the center

	# Apply the transformation to the mesh
	transform = vtk.vtkTransform()
	transform.SetMatrix(roto_translation.ravel())
	transformFilter = vtk.vtkTransformPolyDataFilter()
	transformFilter.SetInputData(mesh)
	transformFilter.SetTransform(transform)
	transformFilter.Update()
	return transformFilter.GetOutput()


def Rototranslate(mesh, roll, pitch, yaw, transl_x, transl_y, transl_z):
	# Rotate the mesh by the angles and translate it
	# The angles are in radians

	rot_z = Rot_z(yaw) # alpha
	rot_y = Rot_y(pitch) # beta
	rot_x = Rot_x(roll) # gamma
	R = np.dot(rot_z, np.dot(rot_y, rot_x))
	rototranslation = np.zeros((4,4))
	rototranslation[:3,:3] = R
	rototranslation[:3,3] = np.dot(-R, np.array([transl_x, transl_y, transl_z]))
	rototranslation[3,3] = 1

	# Apply the transformation to the mesh
	transform = vtk.vtkTransform()
	transform.SetMatrix(rototranslation.ravel())
	transformFilter = vtk.vtkTransformPolyDataFilter()
	transformFilter.SetInputData(mesh)
	transformFilter.SetTransform(transform)
	transformFilter.Update()
	return transformFilter.GetOutput()


def process_kernel(**kwargs):
		
	inputLabelMap = slicer.mrmlScene.GetNodeByID(kwargs['inputLabelMap'])
	outputMesh = slicer.mrmlScene.GetNodeByID(kwargs['outputMesh'])

	target_vertex_count = kwargs['target_vertex_count']
	rot_trans_kwargs = { 'roll': kwargs['roll'], 'pitch': kwargs['pitch'], 'yaw': kwargs['yaw'], 'transl_x': kwargs['transl_x'], 'transl_y': kwargs['transl_y'], 'transl_z': kwargs['transl_z'] }




	"""
		Parameters
	"""

	kwargs = checkInputs(**kwargs)
	
	debug_level = kwargs['debug_level']

	"""
		Beginning of the code
	"""
	debug = DebugLevel(debug_level)

	timings = {}
	timings["Total"] = -time.time()

	
	# get vtk polydata mesh from the input labelmap
	timings["GetMesh"] = -time.time()
	mesh = vtk.vtkPolyData()
	segment_id = inputLabelMap.GetSegmentation().GetSegmentIdBySegmentName("Stone")
	inputLabelMap.GetClosedSurfaceRepresentation(segment_id, mesh)
	timings["GetMesh"] += time.time()


	# get the largest connected component
	timings["LargestConnectedComponent"] = -time.time()
	mesh = select_biggest_connected_component(mesh)
	timings["LargestConnectedComponent"] += time.time()

	# decimate the mesh
	timings["DecimateMesh"] = -time.time()
	decimate_kwargs = {}
	decimate_kwargs['target_reduction'] = (1 - target_vertex_count / mesh.GetNumberOfPoints()) if mesh.GetNumberOfPoints() > target_vertex_count else 0
	mesh = decimate_mesh(mesh, debug, **decimate_kwargs)
	timings["DecimateMesh"] += time.time()

	# Get input directory
	inputDirectory = get_data_directory(inputLabelMap, debug)

	# get the log file information
	log_kwargs = get_scan_info(inputDirectory, debug)

	# Deep copy of the mesh
	mesh_krypton = vtk.vtkPolyData()
	mesh_krypton.DeepCopy(mesh)


	# Roto-translate the mesh
	if log_kwargs is not None:
		timings["Rototranslate"] = -time.time()
		mesh_krypton = CorrectMeshPosition(mesh_krypton, **log_kwargs) # correct the mesh position, positioning it in the center and accounting for the scanning position and the first section
		mesh_krypton = Rototranslate(mesh_krypton, **rot_trans_kwargs) # roto-translate the mesh in the Krypton reference frame
		timings["Rototranslate"] += time.time()

	# put the mesh into the output mesh
	timings["SetMeshAndSave"] = -time.time()
	outputMesh.AddDefaultStorageNode()
	if log_kwargs is not None:
		outputMesh.SetAndObservePolyData(mesh_krypton)
		outputPath = os.path.join(inputDirectory, outputMesh.GetName() +"_krypton" + ".obj")
		outputMesh.GetStorageNode().SetFileName(outputPath)
		outputMesh.GetStorageNode().WriteData(outputMesh)
		logging.info(f"Krypton mesh saved to {outputPath}")
	outputMesh.SetAndObservePolyData(mesh)
	outputPath = os.path.join(inputDirectory, outputMesh.GetName() + ".obj")
	outputMesh.GetStorageNode().SetFileName(outputPath)
	outputMesh.GetStorageNode().WriteData(outputMesh)
	logging.info(f"Mesh saved to {outputPath}")
	timings["SetMeshAndSave"] += time.time()

	timings["Total"] += time.time()


	# print timing
	logging.info("Timings of relevant modules:\n")
	for key, value in timings.items():
		logging.info(f'{value:.2f} s for [{key}]')


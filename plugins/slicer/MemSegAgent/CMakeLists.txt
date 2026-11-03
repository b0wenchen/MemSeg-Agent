set(MODULE_NAME MemSegAgent)
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  MemSegAgentLib/__init__.py
  MemSegAgentLib/client.py
  MemSegAgentLib/volume.py
)
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
)

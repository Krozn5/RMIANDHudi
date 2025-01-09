# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['L2BTree.py'],
             pathex=[],
             binaries=[],
             datas=[('Trained_NN.py', '.'), ('btree.py', '.')],
             hiddenimports=['tensorflow._api.v2.compat.v2.compat.v2.keras.applications.mobilenet', 'tensorflow._api.v2.compat.v1.compat.v1.estimator.experimental', 'tensorflow._api.v2.compat.v1.keras.premade', 'tensorflow._api.v2.compat.v2.keras.applications.inception_v3', 'tensorflow._api.v2.compat.v2.keras.datasets.mnist', 'tensorflow._api.v2.compat.v2.keras.datasets', 'tensorflow._api.v2.compat.v2.keras.applications.resnet', 'tensorflow._api.v2.compat.v1.keras.applications.mobilenet', 'tensorflow._api.v2.compat.v1.keras.__internal__', 'tensorflow._api.v2.compat.v2.keras.preprocessing.text'],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=['gevent', 'greenlet'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,  
          [],
          name='L2BTree',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )

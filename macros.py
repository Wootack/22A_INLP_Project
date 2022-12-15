import numpy as np
from bidict import bidict

###################################################
# Sensors
D001 = np.uint8(0)
D002 = np.uint8(1)
D004 = np.uint8(2)
ENTERHOME = np.uint8(3)
LEAVEHOME = np.uint8(4)
M001 = np.uint8(5)
M002 = np.uint8(6)
M003 = np.uint8(7)
M004 = np.uint8(8)
M005 = np.uint8(9)
M006 = np.uint8(10)
M007 = np.uint8(11)
M008 = np.uint8(12)
M009 = np.uint8(13)
M010 = np.uint8(14)
M011 = np.uint8(15)
M012 = np.uint8(16)
M013 = np.uint8(17)
M014 = np.uint8(18)
M015 = np.uint8(19)
M016 = np.uint8(20)
M017 = np.uint8(21)
M018 = np.uint8(22)
M019 = np.uint8(23)
M020 = np.uint8(24)
M021 = np.uint8(25)
M022 = np.uint8(26)
M023 = np.uint8(27)
M024 = np.uint8(28)
M025 = np.uint8(29)
M026 = np.uint8(30)
M027 = np.uint8(31)
M028 = np.uint8(32)
M029 = np.uint8(33)
M030 = np.uint8(34)
M031 = np.uint8(35)
T001 = np.uint8(36)
T002 = np.uint8(37)
T003 = np.uint8(38)
T004 = np.uint8(39)
T005 = np.uint8(40)
sensor_dict = bidict({ 'D001':D001, 'D002':D002, 'D004':D004, 'ENTERHOME':ENTERHOME, 'LEAVEHOME':LEAVEHOME, 'M001':M001,
                        'M002':M002,'M003':M003,'M004':M004, 'M005':M005, 'M006':M006, 'M007':M007, 'M008':M008, 
                        'M009':M009, 'M010':M010, 'M011':M011, 'M012':M012, 'M013':M013, 'M014':M014, 'M015':M015,
                        'M016':M016, 'M017':M017, 'M018':M018, 'M019':M019, 'M020':M020, 'M021':M021, 'M022':M022,
                        'M023':M023, 'M024':M024, 'M025':M025, 'M026':M026, 'M027':M027, 'M028':M028, 'M029':M029,
                        'M030':M030, 'M031':M031, 'T001':T001, 'T002':T002, 'T003':T003, 'T004':T004, 'T005':T005})


import os


BaseFolder = os.path.join("TSP_Pictures", "RotationTests")
for rotationtests in os.listdir(BaseFolder):
    if os.path.isdir(os.path.join(BaseFolder, rotationtests)):
        rotationtest = os.path.join(BaseFolder, rotationtests)
        for height in os.listdir(rotationtest):
            if os.path.isdir(os.path.join(rotationtest, height)):
                if len(height) > 3:
                    print height
                    heightAddress = os.path.join(rotationtest, height)
                    for Run in os.listdir(heightAddress):
                        if os.path.isdir(os.path.join(heightAddress, Run)):
                            RunAddress = os.path.join(heightAddress, Run)
                            for Type in os.listdir(RunAddress):
                                if os.path.isdir(os.path.join(RunAddress, Type)):
                                    TypeAddress = os.path.join(RunAddress, Type)
                                    for Sign in os.listdir(TypeAddress):
                                        if os.path.isdir(os.path.join(TypeAddress, Sign)):
                                            SignAddress = os.path.join(TypeAddress, Sign)
                                            for internal in os.listdir(SignAddress):
                                                if os.path.isdir(os.path.join(SignAddress, internal)):
                                                    internalAddress = os.path.join(SignAddress, internal)
                                                    number = len([name for name in os.listdir(internalAddress) if os.path.isfile(os.path.join(internalAddress, name))])
                                                    if number != 102:
                                                        print SignAddress, number

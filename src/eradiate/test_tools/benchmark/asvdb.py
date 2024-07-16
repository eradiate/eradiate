# This is a vendored and updated copy of the asvdb project (https://github.com/rapidsai/asvdb).

import glob
import json
import os
import random
import stat
import tempfile
import time
from os import path
from pathlib import Path
from urllib.parse import urlparse

BenchmarkInfoKeys = set(
    [
        "machineName",
        "cudaVer",
        "osType",
        "pythonVer",
        "commitHash",
        "commitTime",
        "branch",
        "gpuType",
        "cpuType",
        "num_cpu",
        "arch",
        "ram",
        "gpuRam",
        "requirements",
        "env_vars",
    ]
)

BenchmarkResultKeys = set(
    [
        "funcName",
        "result",
        "argNameValuePairs",
        "unit",
    ]
)


class BenchmarkInfo:
    """
    Meta-data describing the environment for a benchmark or set of benchmarks.
    """

    def __init__(
        self,
        machineName="",
        cudaVer="",
        osType="",
        pythonVer="",
        commitHash="",
        commitTime=0,
        branch="",
        envName="",
        gpuType="",
        cpuType="",
        numCpu="",
        arch="",
        ram="",
        gpuRam="",
        requirements=None,
        envVar=None,
        resultColumns=None,
    ):
        self.machineName = machineName
        self.cudaVer = cudaVer
        self.osType = osType
        self.gpuType = gpuType
        self.cpuType = cpuType
        self.numCpu = numCpu
        self.arch = arch
        self.ram = ram
        self.gpuRam = gpuRam

        self.pythonVer = pythonVer
        self.commitHash = commitHash
        self.commitTime = int(commitTime)
        self.branch = branch
        self.envName = envName

        self.requirements = requirements or {}
        self.envVar = envVar or {}

        self.resultColumns = resultColumns or []

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(machineName='{self.machineName}'\n"
            f", cudaVer='{self.cudaVer}'\n"
            f", osType='{self.osType}'\n"
            f", pythonVer='{self.pythonVer}'\n"
            f", commitHash='{self.commitHash}'\n"
            f", commitTime={self.commitTime}\n"
            f", branch={self.branch}\n"
            f", envName={self.envName}\n"
            f", gpuType='{self.gpuType}'\n"
            f", cpuType='{self.cpuType}'\n"
            f", numCpu='{self.numCpu}'\n"
            f", arch='{self.arch}'\n"
            f", ram={repr(self.ram)}\n"
            f", gpuRam={repr(self.gpuRam)}\n"
            f", requirements={repr(self.requirements)}\n"
            f", envVar={repr(self.envVar)}\n"
            f", resultColumns={repr(self.resultColumns)}\n"
            ")"
        )

    def __eq__(self, other):
        return (
            (self.machineName == other.machineName)
            and (self.cudaVer == other.cudaVer)
            and (self.osType == other.osType)
            and (self.pythonVer == other.pythonVer)
            and (self.commitHash == other.commitHash)
            and (self.commitTime == other.commitTime)
            and (self.branch == other.branch)
            and (self.envName == other.envName)
            and (self.gpuType == other.gpuType)
            and (self.cpuType == other.cpuType)
            and (self.numCpu == other.numCpu)
            and (self.arch == other.arch)
            and (self.ram == other.ram)
            and (self.gpuRam == other.gpuRam)
            and (self.requirements == other.requirements)
            and (self.envVar == other.envVar)
            and (self.resultColumns == other.resultColumns)
        )


class BenchmarkResult:
    """
    The result of a benchmark run for a particular benchmark function, given
    specific args.
    """

    def __init__(
        self,
        funcName,
        results,
        paramNames=None,
        param=None,
        argNameValuePairs=None,
        unit=None,
        benchType=None,
        code=None,
        version=None,
        minRunCount=2,
        number=0,
        repeat=0,
        rounds=2,
        sampleTime=0.01,
        warmupTime=-1,
    ):
        self.funcName = funcName
        self.argNameValuePairs = self.__sanitizeArgNameValues(argNameValuePairs)

        self.benchType = benchType or "time"
        self.unit = unit or "seconds"
        self.code = code or funcName

        if type(results) is not list:
            results = [results]

        self.paramNames = paramNames or []
        self.param = param or []
        self.version = version or "2"

        self.minRunCount = minRunCount
        self.number = number
        self.repeat = repeat
        self.rounds = rounds
        self.sampleTime = sampleTime
        self.warmupTime = warmupTime

        self.results = results

    def __sanitizeArgNameValues(self, argNameValuePairs):
        if argNameValuePairs is None:
            return []
        return [(n, str(v if v is not None else "NaN")) for (n, v) in argNameValuePairs]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(funcName='{self.funcName}'\n"
            f", results={repr(self.results)}\n"
            f", argNameValuePairs={repr(self.argNameValuePairs)}\n"
            f", unit='{self.unit}'\n"
            f", type='{self.benchType}'\n"
            f", param='{self.param}'\n"
            f", paramNames='{self.paramNames}'\n"
            f", version='{self.version}'\n"
            f", minRunCount='{self.minRunCount}'\n"
            f", number='{self.number}'\n"
            f", repeat='{self.repeat}'\n"
            f", rounds='{self.rounds}'\n"
            f", sampleTime='{self.sampleTime}'\n"
            f", warmupTime='{self.warmupTime}'\n"
            f", results='{self.results}'\n"
            ")"
        )

    def __eq__(self, other):
        return (
            (self.funcName == other.funcName)
            and (self.argNameValuePairs == other.argNameValuePairs)
            and (self.results == other.result)
            and (self.unit == other.unit)
            and (self.code == other.code)
            and (self.param == other.param)
            and (self.paramNames == other.paramNames)
            and (self.version == other.version)
            and (self.minRunCount == other.minRunCount)
            and (self.number == other.number)
            and (self.repeat == other.repeat)
            and (self.rounds == other.rounds)
            and (self.sampleTime == other.sampleTime)
            and (self.warmupTime == other.warmupTime)
            and (self.results == other.results)
        )


class ASVDb:
    """
    A "database" of benchmark results consumable by ASV.
    https://asv.readthedocs.io/en/stable/dev.html?highlight=%24results_dir#benchmark-suite-layout-and-file-formats
    """

    confFileName = "asv.conf.json"
    defaultResultsDirName = "results"
    defaultHtmlDirName = "html"
    defaultConfVersion = 1
    benchmarksFileName = "benchmarks.json"
    machineFileName = "machine.json"
    lockfilePrefix = ".asvdbLOCK"

    def __init__(
        self, dbDir, repo=None, branches=None, projectName=None, commitUrl=None
    ):
        """
        dbDir - directory containing the ASV results, config file, etc.
        repo - the repo associated with all reasults in the DB.
        branches - https://asv.readthedocs.io/en/stable/asv.conf.json.html#branches
        projectName - the name of the project to display in ASV reports
        commitUrl - the URL ASV will use in reports to redirect users to when
                    they click on a data point. This is typically a Github
                    project URL that shows the contents of a commit.
        """
        self.dbDir = dbDir
        self.repo = repo
        self.branches = branches
        self.projectName = projectName
        self.commitUrl = commitUrl

        self.machineFileExt = path.join(
            self.defaultResultsDirName, "*", self.machineFileName
        )
        self.confFileExt = self.confFileName
        self.confFilePath = path.join(self.dbDir, self.confFileName)
        self.confVersion = self.defaultConfVersion
        self.resultsDirName = self.defaultResultsDirName
        self.resultsDirPath = path.join(self.dbDir, self.resultsDirName)
        self.htmlDirName = self.defaultHtmlDirName
        self.benchmarksFileExt = path.join(
            self.defaultResultsDirName, self.benchmarksFileName
        )
        self.benchmarksFilePath = path.join(
            self.resultsDirPath, self.benchmarksFileName
        )

        # Each ASVDb instance must have a unique lockfile name to identify other
        # instances that may be setting locks.
        self.lockfileName = "%s-%s-%s" % (self.lockfilePrefix, os.getpid(), time.time())
        self.lockfileTimeout = 5  # seconds

        # S3-related attributes
        if self.__isS3URL(dbDir):
            self.s3Resource = None  # boto3.resource("s3")
            self.bucketName = urlparse(self.dbDir, allow_fragments=False).netloc
            self.bucketKey = urlparse(self.dbDir, allow_fragments=False).path.lstrip(
                "/"
            )

        ########################################
        # Testing and debug members
        self.debugPrint = False
        # adds a delay during write operations to easily test write collision
        # handling.
        self.writeDelay = 0
        # To "cancel" write operations that are being delayed.
        self.cancelWrite = False

    ###########################################################################
    # Public API
    ###########################################################################
    def loadConfFile(self):
        """
        Read the ASV conf file on disk and set - or possibly overwrite - the
        member variables with the contents of the file.
        """
        self.__assertDbDirExists()
        try:
            self.__getLock(self.dbDir)
            # FIXME: check if confFile exists
            self.__downloadIfS3()

            d = self.__loadJsonDictFromFile(self.confFilePath)
            self.resultsDirName = d.get("results_dir", self.resultsDirName)
            self.resultsDirPath = path.join(self.dbDir, self.resultsDirName)
            self.benchmarksFilePath = path.join(
                self.resultsDirPath, self.benchmarksFileName
            )
            self.htmlDirName = d.get("html_dir", self.htmlDirName)
            self.repo = d.get("repo")
            self.branches = d.get("branches", [])
            self.projectName = d.get("project")
            self.commitUrl = d.get("show_commit_url")

            self.__uploadIfS3()

        finally:
            self.__releaseLock(self.dbDir)
            self.__removeLocalS3Copy()

    def updateConfFile(self):
        """
        Update the ASV conf file with the values passed in to the CTOR. This
        also ensures the object is up-to-date with any changes to the conf file
        that may have been done by other ASVDb instances.
        """
        self.__ensureDbDirExists()
        try:
            self.__getLock(self.dbDir)
            self.__downloadIfS3()

            if self.__waitForWrite():
                self.__updateConfFile()

            self.__uploadIfS3()

        finally:
            self.__releaseLock(self.dbDir)
            self.__removeLocalS3Copy()

    def addResult(self, benchmarkInfo, benchmarkResult):
        """
        Add the benchmarkResult associated with the benchmarkInfo to the DB.
        This will also update the conf file with the CTOR args if not done
        already.
        """
        self.__ensureDbDirExists()
        try:
            self.__getLock(self.dbDir)
            self.__downloadIfS3(bInfo=benchmarkInfo)
            if self.__waitForWrite():
                self.__updateFilesForInfo(benchmarkInfo)
                self.__updateFilesForResult(benchmarkInfo, benchmarkResult)

            self.__uploadIfS3()

        finally:
            self.__releaseLock(self.dbDir)
            self.__removeLocalS3Copy()

    def addResults(self, benchmarkInfo, benchmarkResultList):
        """
        Add each benchmarkResult obj in benchmarkResultList associated with
        benchmarkInfo to the DB.  This will also update the conf file with the
        CTOR args if not done already.
        """
        self.__ensureDbDirExists()
        try:
            self.__getLock(self.dbDir)
            self.__downloadIfS3(bInfo=benchmarkInfo)

            if self.__waitForWrite():
                self.__updateFilesForInfo(benchmarkInfo)
                for resultObj in benchmarkResultList:
                    self.__updateFilesForResult(benchmarkInfo, resultObj)

            self.__uploadIfS3()

        finally:
            self.__releaseLock(self.dbDir)
            self.__removeLocalS3Copy()

    def getInfo(self):
        """
        Return a list of BenchmarkInfo objs from reading the db files on disk.
        """
        self.__assertDbDirExists()
        try:
            self.__getLock(self.dbDir)
            self.__downloadIfS3()
            retList = self.__readResults(infoOnly=True)

        finally:
            self.__releaseLock(self.dbDir)
            self.__removeLocalS3Copy()

        return retList

    def getResults(self, filterInfoObjList=None):
        """
        Return a list of (BenchmarkInfo obj, [BenchmarkResult obj, ...]) tuples
        from reading the db files on disk.  filterInfoObjList is expected to be
        a list of BenchmarkInfo objs, and if provided will be used to return
        results for only those BenchmarkInfo objs.
        """
        self.__assertDbDirExists()
        try:
            self.__getLock(self.dbDir)
            self.__downloadIfS3(results=True)
            retList = self.__readResults(filterByInfoObjs=filterInfoObjList)

        finally:
            self.__releaseLock(self.dbDir)
            self.__removeLocalS3Copy()

        return retList

    ###########################################################################
    # Private methods. These should not be called by clients. Among other
    # things, public methods use proper locking to ensure atomic operations
    # and these do not.
    ###########################################################################
    def __readResults(self, infoOnly=False, filterByInfoObjs=None):
        """
        Main "read" method responsible for reading ASV JSON files and creating
        BenchmarkInfo and BenchmarkResult objs.

        If infoOnly==True, returns a list of only BenchmarkInfo objs, otherwise
        returns a list of tuples containing (BenchmarkInfo obj, [BenchmarkResult
        obj, ...]) to represent each BenchmarkInfo object and all the
        BenchmarkResult objs associated with it.

        filterByInfoObjs can be set to only return BenchmarkInfo objs and their
        results that match at least one of the BenchmarkInfo objs in the
        filterByInfoObjs list (the list is treated as ORd).
        """
        retList = []

        resultsPath = Path(self.resultsDirPath)

        # benchmarks.json containes meta-data about the individual benchmarks,
        # which is only needed for returning results.
        if not (infoOnly):
            benchmarksJsonFile = resultsPath / self.benchmarksFileName
            if benchmarksJsonFile.exists():
                bDict = self.__loadJsonDictFromFile(benchmarksJsonFile.as_posix())
            else:
                # FIXME: test
                raise FileNotFoundError(f"{benchmarksJsonFile.as_posix()}")

        for machineDir in resultsPath.iterdir():
            # Each subdir under the results dir contains all results for a
            # individual machine. The only non-dir (file) that may need to be
            # read in the results dir is benchmarks.json, which would have been
            # read above.
            if machineDir.is_dir():
                # Inside the individual machine dir, ;ook for and read
                # machine.json first.  Assume this is not a valid results dir if
                # no machine file and skip.
                machineJsonFile = machineDir / self.machineFileName
                if machineJsonFile.exists():
                    mDict = self.__loadJsonDictFromFile(machineJsonFile.as_posix())
                else:
                    continue

                # Read each results file and populate the machineResults list.
                # This will contain either BenchmarkInfo objs or tuples of
                # (BenchmarkInfo, [BenchmarkResult objs, ...]) based on infoOnly
                machineResults = []
                for resultsFile in machineDir.iterdir():
                    if resultsFile == machineJsonFile:
                        continue
                    rDict = self.__loadJsonDictFromFile(resultsFile.as_posix())

                    resultsParams = rDict.get("params", {})
                    # Each results file has a single BenchmarkInfo obj
                    # describing it.
                    bi = BenchmarkInfo(
                        machineName=mDict.get("machine", ""),
                        cudaVer=resultsParams.get("cuda", ""),
                        osType=resultsParams.get("os", ""),
                        pythonVer=resultsParams.get("python", ""),
                        commitHash=rDict.get("commit_hash", ""),
                        commitTime=rDict.get("date", ""),
                        branch=rDict.get("branch", ""),
                        envName=rDict.get("env_name", ""),
                        gpuType=mDict.get("gpu", ""),
                        cpuType=mDict.get("cpu", ""),
                        numCpu=mDict.get("num_cpu", ""),
                        arch=mDict.get("arch", ""),
                        ram=mDict.get("ram", ""),
                        gpuRam=mDict.get("gpuRam", ""),
                        requirements=rDict.get("requirements", {}),
                        envVar=rDict.get("env_vars", {}),
                        resultColumns=rDict.get("result_columns", None),
                    )

                    # If a filter was specified, at least one EXACT MATCH to the
                    # BenchmarkInfo obj must be present.
                    if filterByInfoObjs and bi not in filterByInfoObjs:
                        continue

                    if infoOnly:
                        machineResults.append(bi)
                    else:
                        # FIXME: if results not in rDict, throw better error
                        resultsDict = rDict["results"]
                        # Populate the list of BenchmarkResult objs associated
                        # with the BenchmarkInfo obj
                        resultObjs = []
                        for benchmarkName in resultsDict:
                            # benchmarkSpec is the entry in benchmarks.json,
                            # which is needed for the param names
                            if benchmarkName not in bDict:
                                print(
                                    "WARNING: Encountered benchmark name "
                                    "that is not in "
                                    f"{self.benchmarksFileName}: "
                                    f"file: {resultsFile.as_posix()} "
                                    f'invalid name"{benchmarkName}", skipping.'
                                )
                                continue

                            # the number of result columns should always be the same as the number of results
                            # not true for because samples and profile only exist if the benchmark is run with profile mode
                            # assert len(bi.resultColumns) == len(resultsDict[benchmarkName])

                            benchmarkSpec = bDict[benchmarkName]
                            # benchmarkResults is the entry in this particular
                            # result file for this benchmark
                            # benchmarkResults = dict(zip(rDict["result_columns"],resultsDict[benchmarkName]))

                            # dropping the parameter picking. we will just overwrite
                            br = BenchmarkResult(
                                funcName=benchmarkName,
                                results=resultsDict[benchmarkName],
                                paramNames=benchmarkSpec["param_names"],
                                param=benchmarkSpec["params"],
                                code=benchmarkSpec["code"],
                                version=benchmarkSpec["version"],
                                benchType=benchmarkSpec["type"],
                                unit=benchmarkSpec["unit"],
                                minRunCount=benchmarkSpec["min_run_count"],
                                number=benchmarkSpec["number"],
                                repeat=benchmarkSpec["repeat"],
                                rounds=benchmarkSpec["rounds"],
                                sampleTime=benchmarkSpec["sample_time"],
                                warmupTime=benchmarkSpec["warmup_time"],
                            )
                            unit = benchmarkSpec.get("unit")
                            code = benchmarkSpec.get("code")
                            if unit is not None:
                                br.unit = unit
                            if code is not None:
                                br.code = code
                            resultObjs.append(br)

                            # NM 18/07/2024: Disabled parameter expansion
                            # -------------------------------------------

                            # Inverse of the write operation described in
                            # self.__updateResultJson()
                            # paramsCartProd = list(itertools.product(*paramValues))
                            # for (paramValueCombo, result) in zip(paramsCartProd, results):
                            # br = BenchmarkResult(
                            # funcName=benchmarkName,
                            # argNameValuePairs=zip(paramNames, paramValueCombo),
                            # result=result)
                            #     unit = benchmarkSpec.get("unit")
                            #     if unit is not None:
                            #         br.unit = unit
                            #     resultObjs.append(br)
                        machineResults.append((bi, resultObjs))

                retList += machineResults

        return retList

    def __updateFilesForInfo(self, benchmarkInfo):
        """
        Updates all the db files that are affected by a new BenchmarkInfo obj.
        """
        # special case: if the benchmarkInfo has a new branch specified,
        # update self.branches so the conf files includes the new branch
        # name.
        newBranch = benchmarkInfo.branch
        if newBranch and newBranch not in self.branches:
            self.branches.append(newBranch)

        # The comments below assume default dirname values (mainly
        # "results"), which can be changed in the asv.conf.json file.
        #
        # <self.dbDir>/asv.conf.json
        self.__updateConfFile()
        # <self.dbDir>/results/<machine dir>/machine.json
        self.__updateMachineJson(benchmarkInfo)

    def __updateFilesForResult(self, benchmarkInfo, benchmarkResult):
        """
        Updates all the db files that are affected by a new BenchmarkResult
        obj. This also requires the corresponding BenchmarkInfo obj since some
        results files also include info data.
        """
        # <self.dbDir>/results/benchmarks.json
        self.__updateBenchmarkJson(benchmarkResult)
        # <self.dbDir>/results/<machine dir>/<result file name>.json
        self.__updateResultJson(benchmarkResult, benchmarkInfo)

    def __assertDbDirExists(self):
        # FIXME: update for support S3 - this method should return True if
        # self.dbDir is a valid S3 URL or a valid path on disk.
        if self.__isS3URL(self.dbDir):
            self.s3Resource.Bucket(self.bucketName).objects
        else:
            if not (path.isdir(self.dbDir)):
                raise FileNotFoundError(
                    f"{self.dbDir} does not exist or is " "not a directory"
                )

    def __ensureDbDirExists(self):
        # FIXME: for S3 support, if self.dbDir is a S3 URL then simply check if
        # it's valid and exists, but don't try to create it (raise an exception
        # if it does not exist).  For a local file path, create it if it does
        # not exist, like already being done below.
        if self.__isS3URL(self.dbDir):
            self.s3Resource.Bucket(self.bucketName).objects
        else:
            if not (path.exists(self.dbDir)):
                os.mkdir(self.dbDir)
                # Hack: os.mkdir() seems to return before the filesystem catches up,
                # so pause before returning to help ensure the dir actually exists
                time.sleep(0.1)

    def __updateConfFile(self):
        """
        Update the conf file with the settings in this ASVDb instance.
        """
        if self.repo is None:
            raise AttributeError(
                "repo must be set to non-None before " f"writing {self.confFilePath}"
            )

        d = self.__loadJsonDictFromFile(self.confFilePath)
        # ASVDb is git-only for now, so ensure .git extension
        d["repo"] = self.repo + (".git" if not self.repo.endswith(".git") else "")
        currentBranches = d.get("branches", [])
        d["branches"] = currentBranches + [
            b for b in (self.branches or []) if b not in currentBranches
        ]
        d["version"] = self.confVersion
        d["project"] = self.projectName or self.repo.replace(".git", "").split("/")[-1]
        d["show_commit_url"] = self.commitUrl or (
            self.repo.replace(".git", "")
            + ("/" if not self.repo.endswith("/") else "")
            + "commit/"
        )

        self.__writeJsonDictToFile(d, self.confFilePath)

    def __updateBenchmarkJson(self, benchmarkResult):
        # The following is an example of the schema ASV expects for
        # `benchmarks.json`.  If param names are A, B, and C
        #
        # {
        #     "<algo name>": {
        #         "code": "",
        #         "name": "<algo name>",
        #         "param_names": [
        #             "A", "B", "C"
        #         ],
        #     "params": [
        #                [<value1 for A>,
        #                 <value2 for A>,
        #                ],
        #                [<value1 for B>,
        #                 <value2 for B>,
        #                ],
        #                [<value1 for C>,
        #                 <value2 for C>,
        #                ],
        #               ],
        #         "timeout": 60,
        #         "type": "time",
        #         "unit": "seconds",
        #         "version": 1,
        #     }
        # }

        newParamNames = []
        newParamValues = []
        for n, v in benchmarkResult.argNameValuePairs:
            newParamNames.append(n)
            newParamValues.append(v)

        d = self.__loadJsonDictFromFile(self.benchmarksFilePath)

        benchDict = d.setdefault(
            benchmarkResult.funcName,
            self.__getDefaultBenchmarkDescrDict(
                benchmarkResult.funcName, newParamNames
            ),
        )
        benchDict["unit"] = benchmarkResult.unit
        benchDict["type"] = benchmarkResult.benchType
        benchDict["code"] = benchmarkResult.code
        benchDict["version"] = benchmarkResult.version

        benchDict["param_names"] = benchmarkResult.paramNames
        benchDict["params"] = benchmarkResult.param

        benchDict["min_run_count"] = benchmarkResult.minRunCount
        benchDict["number"] = benchmarkResult.number
        benchDict["repeat"] = benchmarkResult.repeat
        benchDict["rounds"] = benchmarkResult.rounds
        benchDict["sample_time"] = benchmarkResult.sampleTime
        benchDict["warmup_time"] = benchmarkResult.warmupTime

        # NM 18/07/2024: Disabled parameter expansion
        # -------------------------------------------

        # existingParamNames = benchDict["param_names"]
        # existingParamValues = benchDict["params"]

        # numExistingParams = len(existingParamNames)
        # numExistingParamValues = len(existingParamValues)
        # numNewParams = len(newParamNames)

        # # Check for the case where a result came in for the function, but it has
        # # a different number of args vs. what was saved previously
        # if numExistingParams != numNewParams:
        #     raise ValueError("result for %s had %d params in benchmarks.json, "
        #                      "but new result has %d params" \
        #                      % (benchmarkResult.funcName, numExistingParams,
        #                         numNewParams))
        # numParams = numNewParams

        # cartProd = list(itertools.product(*existingParamValues))
        # if tuple(newParamValues) not in cartProd:
        #     if numExistingParamValues == 0:
        #         for newVal in newParamValues:
        #             existingParamValues.append([newVal])
        #     else:
        #         for i in range(numParams):
        #             if newParamValues[i] not in existingParamValues[i]:
        #                 existingParamValues[i].append(newParamValues[i])

        d[benchmarkResult.funcName] = benchDict

        # a version key must always be present in self.benchmarksFilePath,
        # "current" ASV version requires this to be 2 (or higher?)
        # d["version"] = benchmarkResult["version"]
        d["version"] = 2
        self.__writeJsonDictToFile(d, self.benchmarksFilePath)

    def __updateMachineJson(self, benchmarkInfo):
        # The following is an example of the schema ASV expects for
        # `machine.json`.
        # {
        #     "arch": "x86_64",
        #     "cpu": "Intel, ...",
        #     "machine": "sm01",
        #     "os": "Linux ...",
        #     "ram": "123456",
        #     "version": 1,
        # }

        machineFilePath = path.join(
            self.resultsDirPath, benchmarkInfo.machineName, self.machineFileName
        )
        d = self.__loadJsonDictFromFile(machineFilePath)
        d["arch"] = benchmarkInfo.arch
        d["cpu"] = benchmarkInfo.cpuType
        d["gpu"] = benchmarkInfo.gpuType
        # d["cuda"] = benchmarkInfo.cudaVer
        d["machine"] = benchmarkInfo.machineName
        # d["os"] = benchmarkInfo.osType
        d["ram"] = benchmarkInfo.ram
        d["gpuRam"] = benchmarkInfo.gpuRam
        d["version"] = 1
        self.__writeJsonDictToFile(d, machineFilePath)

    def __updateResultJson(self, benchmarkResult, benchmarkInfo):
        # The following is an example of the schema ASV expects for
        # '<machine>-<commit_hash>.json'. If param names are A, B, and C
        #
        # {
        #     "params": {
        #         "cuda": "9.2",
        #         "gpu": "Tesla ...",
        #         "machine": "sm01",
        #         "os": "Linux ...",
        #         "python": "3.7",
        #     },
        #     "requirements": {},
        #     "result_columns":["result", "params", "version", "started_at", "duration", "stats_ci_99_a", "stats_ci_99_b", "stats_q_25", "stats_q_75", "stats_number", "stats_repeat", "samples", "profile"]
        #     "results": {
        #         "<algo name>": {
        #              [[<result1>,<result2>,],[<param1>, <param2>],"<version>",...]
        #     },
        #     "commit_hash": "321e321321eaf",
        #     "date": 12345678,
        #     "python": "3.7",
        #     "version": 1,
        # }

        resultsFilePath = self.__getResultsFilePath(benchmarkInfo)
        d = self.__loadJsonDictFromFile(resultsFilePath)

        d["commit_hash"] = benchmarkInfo.commitHash
        d["env_name"] = benchmarkInfo.envName
        d["date"] = int(benchmarkInfo.commitTime)
        d["python"] = benchmarkInfo.pythonVer
        d["params"] = {
            "arch": benchmarkInfo.arch,
            "cpu": benchmarkInfo.cpuType,
            "num_cpu": benchmarkInfo.numCpu,
            "machine": benchmarkInfo.machineName,
            "os": benchmarkInfo.osType,
            "python": benchmarkInfo.pythonVer,
        }
        d["requirements"] = benchmarkInfo.requirements
        d["env_vars"] = benchmarkInfo.envVar
        d["result_columns"] = benchmarkInfo.resultColumns

        allResultsDict = d.setdefault("results", {})
        allResultsDict[benchmarkResult.funcName] = benchmarkResult.results
        d["version"] = 2

        # NM 18/07/2024: Disabled parameter expansion
        # -------------------------------------------

        # resultDict = allResultsDict.setdefault(benchmarkResult.funcName, {})

        # existingParamValuesList = benchmarkInfo.params
        # existingResultValueList = resultDict.setdefault("result", [])

        # ASV uses the cartesian product of the param values for looking up the
        # result for a particular combination of param values.  For example:
        # "params": [["a"], ["b", "c"], ["d", "e"]] results in: [("a", "b",
        # "d"), ("a", "b", "e"), ("a", "c", "d"), ("a", "c", "e")] and each
        # combination of param values has a result, with the results for the
        # corresponding param values in the same order.  If a result for a set
        # of param values DNE, use None.

        # store existing results in map based on cartesian product of all
        # current params.
        # paramsCartProd = list(itertools.product(*existingParamValuesList))
        # Assume there is an equal number of results for cartProd values
        # (some will be None)
        # paramsResultMap = dict(zip(paramsCartProd, existingResultValueList))

        # FIXME: dont assume these are ordered properly (ie. the same way as
        # defined in benchmarks.json)
        # newResultParamValues = tuple(v for (_, v) in benchmarkResult.argNameValuePairs)

        # Update the "params" lists with the new param settings for the new result.
        # Only add values that are not already present
        # numExistingParamValues = len(existingParamValuesList)
        # if numExistingParamValues == 0:
        #     for newParamValue in newResultParamValues:
        #         existingParamValuesList.append([newParamValue])
        #     results = [benchmarkResult.results]

        # else:
        #     for i in range(numExistingParamValues):
        #         if newResultParamValues[i] not in existingParamValuesList[i]:
        #             existingParamValuesList[i].append(newResultParamValues[i])

        #     # Add the new result
        #     paramsResultMap[newResultParamValues] = benchmarkResult.results

        #     # Re-compute the cartesian product of all param values now that the
        #     # new values are added. Use this to determine where to place the new
        #     # result in the result list.
        #     results = []
        #     for paramVals in itertools.product(*existingParamValuesList):
        #         results.append(paramsResultMap.get(paramVals))

        # resultDict["params"] = existingParamValuesList
        # resultDict["results"] = benchmarkResult.results

        self.__writeJsonDictToFile(d, resultsFilePath)

    def __getDefaultBenchmarkDescrDict(self, funcName, paramNames):
        return {
            "code": funcName,
            "min_run_count": 2,
            "name": funcName,
            "number": 0,
            "param_names": paramNames,
            "params": [],
            "repeat": 0,
            "rounds": 1,
            "sample_time": 0.01,
            "type": "time",
            "unit": "seconds",
            "version": 2,
            "warmup_time": -1,
        }

    def __getResultsFilePath(self, benchmarkInfo):
        # The path to the resultsFile will be based on additional params present
        # in the benchmarkInfo obj.
        assert (
            len(benchmarkInfo.commitHash[:8]) >= 8
        ), "commit hash needs to be at leas 8 characters long"

        fileNameParts = [
            benchmarkInfo.commitHash[:8],
            benchmarkInfo.envName,
        ]
        fileName = "-".join(fileNameParts) + ".json"
        return path.join(self.resultsDirPath, benchmarkInfo.machineName, fileName)

    def __loadJsonDictFromFile(self, jsonFile):
        """
        Return a dictionary representing the contents of jsonFile by
        either reading in the existing file or returning {}
        """
        if path.exists(jsonFile):
            with open(jsonFile) as fobj:
                # FIXME: ideally this could use flock(), but some situations do
                # not allow grabbing a file lock (NFS?)
                # fcntl.flock(fobj, fcntl.LOCK_EX)
                # FIXME: error checking
                return json.load(fobj)

        return {}

    def __writeJsonDictToFile(self, jsonDict, filePath):
        # FIXME: error checking
        dirPath = path.dirname(filePath)
        if not path.isdir(dirPath):
            os.makedirs(dirPath)

        with open(filePath, "w") as fobj:
            # FIXME: ideally this could use flock(), but some situations do not
            # allow grabbing a file lock (NFS?)
            # fcntl.flock(fobj, fcntl.LOCK_EX)
            json.dump(jsonDict, fobj, indent=2)

    ###########################################################################
    # ASVDb private locking methods
    ###########################################################################
    def __getLock(self, dirPath):
        if self.__isS3URL(dirPath):
            self.__getS3Lock()
        else:
            self.__getLocalFileLock(dirPath)

    def __getLocalFileLock(self, dirPath):
        """
        Gets a lock on dirPath against other ASVDb instances (in other
        processes, possibily on other machines) using the following technique:

        * Check for other locks and clear them if they've been seen for longer
          than self.lockfileTimeout (do this to help cleanup after others that
          may have died prematurely)

        * Once all locks are clear - either by their owner because they
          finished their read/write, or by removing them because they're
          presumed dead - create a lock for this instance

        * If a race condition was detected, probably because multiple ASVDbs
          saw all locks were cleared at the same time and created their locks
          at the same time, remove this lock, and wait a random amount of time
          before trying again.  The random time prevents yet another race.
        """
        otherLockfileTimes = {}
        thisLockfile = path.join(dirPath, self.lockfileName)
        # FIXME: This shouldn't be needed? But if so, be smarter about
        # preventing an infintite loop?
        i = 0
        while i < 1000:
            # Keep checking for other locks to clear
            self.__updateOtherLockfileTimes(dirPath, otherLockfileTimes)
            # FIXME: potential infintite loop due to starvation?
            otherLocks = list(otherLockfileTimes.keys())
            while otherLocks:
                if self.debugPrint:
                    print(
                        f"This lock file will be {thisLockfile} but other "
                        f"locks present: {otherLocks}, waiting to try to "
                        "lock again..."
                    )
                time.sleep(0.2)
                self.__updateOtherLockfileTimes(dirPath, otherLockfileTimes)
                otherLocks = list(otherLockfileTimes.keys())

            # All clear, create lock
            if self.debugPrint:
                print(f"All clear, setting lock {thisLockfile}")
            self.__createLockfile(dirPath)

            # Check for a race condition where another lock could have been created
            # while creating the lock for this instance.
            self.__updateOtherLockfileTimes(dirPath, otherLockfileTimes)

            # If another lock snuck in while this instance was creating its
            # lock, remove this lock and wait a random amount of time before
            # trying again (random time to prevent another race condition with
            # the competing instance, this way someone will clearly get there
            # first)
            if otherLockfileTimes:
                self.__releaseLock(dirPath)
                randTime = (int(5 * random.random()) + 1) + random.random()
                if self.debugPrint:
                    print(
                        f"Collision - waiting {randTime} seconds before "
                        "trying to lock again."
                    )
                time.sleep(randTime)
            else:
                break

            i += 1

    def __releaseLock(self, dirPath):
        if self.__isS3URL(dirPath):
            self.__releaseS3Lock()
        else:
            self.__releaseLocalFileLock(dirPath)

    def __releaseLocalFileLock(self, dirPath):
        thisLockfile = path.join(dirPath, self.lockfileName)
        if self.debugPrint:
            print(f"Removing lock {thisLockfile}")
        self.__removeFiles([thisLockfile])

    def __updateOtherLockfileTimes(self, dirPath, lockfileTimes):
        """
        Return a list of lockfiles that have "timed out", probably because their
        process was killed. This will never include the lockfile for this
        instance.  Update the lockfileTimes dict as a side effect with the
        discovery time of any new lockfiles and remove any lockfiles that are no
        longer present.
        """
        thisLockfile = path.join(dirPath, self.lockfileName)
        now = time.time()
        expired = []

        allLockfiles = glob.glob(path.join(dirPath, self.lockfilePrefix) + "*")

        if self.debugPrint:
            print(
                f"   This lockfile is {thisLockfile}, allLockfiles is "
                f"{allLockfiles}, lockfileTimes is {lockfileTimes}"
            )
        # Remove lockfiles from the lockfileTimes dict that are no longer
        # present on disk
        lockfilesToRemove = set(lockfileTimes.keys()) - set(allLockfiles)
        for removedLockfile in lockfilesToRemove:
            lockfileTimes.pop(removedLockfile)

        # check for expired lockfiles while also setting the discovery time on
        # new lockfiles in the lockfileTimes dict.
        for lockfile in allLockfiles:
            if lockfile == thisLockfile:
                continue
            if (now - lockfileTimes.setdefault(lockfile, now)) > self.lockfileTimeout:
                expired.append(lockfile)

        if self.debugPrint:
            print(
                f"   This lockfile is {thisLockfile}, lockfileTimes is "
                f"{lockfileTimes}, now is {now}, expired is {expired}"
            )
        self.__removeFiles(expired)

    def __createLockfile(self, dirPath):
        """
        low-level lockfile creation - consider calling __getLock() instead.
        """
        thisLockfile = path.join(dirPath, self.lockfileName)
        open(thisLockfile, "w").close()
        # Make the lockfile read/write to all so others can remove it if this
        # process dies prematurely
        os.chmod(
            thisLockfile,
            (
                stat.S_IRUSR
                | stat.S_IWUSR
                | stat.S_IRGRP
                | stat.S_IWGRP
                | stat.S_IROTH
                | stat.S_IWOTH
            ),
        )

    ###########################################################################
    # S3 Locking methods
    ###########################################################################
    def __getS3Lock(self):
        thisLockfile = path.join(self.bucketKey, self.lockfileName)
        # FIXME: This shouldn't be needed? But if so, be smarter about
        # preventing an infintite loop?
        i = 0

        # otherLockfileTimes is a tuple representing (<List of lockfiles>, <Length of List>)
        otherLockfileTimes = ([], 0)
        while i < 1000:
            otherLockfileTimes = self.__updateS3LockfileTimes()
            debugCounter = 0

            while otherLockfileTimes[1] != 0:
                if self.debugPrint:
                    lockfileList = []
                    for each in otherLockfileTimes[0]:
                        lockfileList.append(each.key)

                    print(
                        f"This lock file will be {thisLockfile} but other "
                        f"locks present: {lockfileList}, waiting to try to "
                        "lock again..."
                    )

                time.sleep(1)
                otherLockfileTimes = self.__updateS3LockfileTimes()

            # All clear, create lock
            if self.debugPrint:
                print(f"All clear, setting lock {thisLockfile}")
            self.s3Resource.Object(self.bucketName, thisLockfile).put()

            # Give S3 time to see the new lock
            time.sleep(1)

            # Check for a race condition where another lock could have been created
            # while creating the lock for this instance.
            otherLockfileTimes = self.__updateS3LockfileTimes()

            if otherLockfileTimes[1] != 0:
                self.__releaseS3Lock()
                randTime = (int(30 * random.random()) + 5) + random.random()
                if self.debugPrint:
                    print(
                        f"Collision - waiting {randTime} seconds before "
                        "trying to lock again."
                    )
                time.sleep(randTime)
            else:
                break

            i += 1

    def __updateS3LockfileTimes(self):
        # Find lockfiles in S3 Bucket
        response = self.s3Resource.Bucket(self.bucketName).objects.filter(
            Prefix=path.join(self.bucketKey, self.lockfilePrefix)
        )

        length = 0
        for lockfile in response:
            length += 1
            if self.lockfileName in lockfile.key:
                lockfile.delete()
                length -= 1

        return (response, length)

    def __releaseS3Lock(self):
        thisLockfile = path.join(self.bucketKey, self.lockfileName)
        if self.debugPrint:
            print(f"Removing lock {thisLockfile}")
        self.s3Resource.Object(self.bucketName, thisLockfile).delete()

    ###########################################################################
    # S3 utilities
    ###########################################################################
    def __downloadIfS3(self, bInfo=BenchmarkInfo(), results=False):
        def downloadS3(bucket, ext):
            bucket.download_file(
                path.join(self.bucketKey, ext), path.join(self.localS3Copy.name, ext)
            )

        if not self.__isS3URL(self.dbDir):
            return

        self.localS3Copy = tempfile.TemporaryDirectory()
        os.makedirs(path.join(self.localS3Copy.name, self.defaultResultsDirName))
        bucket = self.s3Resource.Bucket(self.bucketName)
        # If results isn't set, only download key files, else download key files and results
        if results == False:
            keyFileExts = [
                self.confFileExt,
                self.machineFileExt,
                self.benchmarksFileExt,
            ]
            # Use Try/Except to catch file Not Found errors and continue, avoids additional API calls
            for fileExt in keyFileExts:
                try:
                    downloadS3(bucket, fileExt)
                except Exception as e:
                    err = "Not Found"
                    if err not in e.response["Error"]["Message"]:
                        raise

            # Download specific result file for updating results if BenchmarkInfo is sent
            try:
                if bInfo.machineName != "":
                    commitHash, pyVer, cuVer, osType = (
                        bInfo.commitHash,
                        bInfo.pythonVer,
                        bInfo.cudaVer,
                        bInfo.osType,
                    )
                    filename = f"{commitHash}-python{pyVer}-cuda{cuVer}-{osType}.json"
                    os.makedirs(
                        path.join(
                            self.localS3Copy.name,
                            self.defaultResultsDirName,
                            bInfo.machineName,
                        ),
                        exist_ok=True,
                    )
                    resultFileExt = path.join(
                        self.defaultResultsDirName, bInfo.machineName, filename
                    )
                    downloadS3(bucket, resultFileExt)

            except Exception as e:
                err = "Not Found"
                if err not in e.response["Error"]["Message"]:
                    raise

        else:
            try:
                downloadS3(bucket, self.confFileExt)
            except Exception as e:
                err = "Not Found"
                if err not in e.response["Error"]["Message"]:
                    raise

            try:
                resultsBucketPath = path.join(
                    self.bucketKey, self.defaultResultsDirName
                )
                resultsLocalPath = path.join(
                    self.localS3Copy.name, self.defaultResultsDirName
                )

                # Loop over ASV results folder and download everything.
                # objectExt represents the file extension starting from the base resultsBucketPath
                # For example: resultsBucketPath = "asvdb/results"
                #            : objectKey = "asvdb/results/machine_name/results.json
                #            : objectExt = "machine_name/results.json"
                for bucketObj in bucket.objects.filter(Prefix=resultsBucketPath):
                    objectExt = bucketObj.key.replace(resultsBucketPath + "/", "")
                    if len(objectExt.split("/")) > 1:
                        os.makedirs(
                            path.join(resultsLocalPath, objectExt.split("/")[0]),
                            exist_ok=True,
                        )
                    bucket.download_file(
                        bucketObj.key, path.join(resultsLocalPath, objectExt)
                    )

            except Exception as e:
                err = "Not Found"
                if err not in e.response["Error"]["Message"]:
                    raise e

        # Set all the internal locations to point to the downloaded files:
        self.confFilePath = path.join(self.localS3Copy.name, self.confFileName)
        self.resultsDirPath = path.join(self.localS3Copy.name, self.resultsDirName)
        self.benchmarksFilePath = path.join(
            self.resultsDirPath, self.benchmarksFileName
        )

    def __uploadIfS3(self):
        def recursiveUpload(base, ext=""):
            root, dirs, files = next(os.walk(path.join(base, ext), topdown=True))

            # Upload files in this folder
            for name in files:
                self.s3Resource.Bucket(self.bucketName).upload_file(
                    path.join(base, ext, name), path.join(self.bucketKey, ext, name)
                )

            # Call upload again for each folder
            if len(dirs) != 0:
                for folder in dirs:
                    ext = path.join(ext, folder)
                    recursiveUpload(base, ext)

        if self.__isS3URL(self.dbDir):
            recursiveUpload(self.localS3Copy.name)
            # Give S3 time to see the new uploads before releasing lock
            time.sleep(1)

    def __removeLocalS3Copy(self):
        if not self.__isS3URL(self.dbDir):
            return

        self.localS3Copy.cleanup()
        self.localS3Copy = None

        self.confFilePath = path.join(self.dbDir, self.confFileName)
        self.resultsDirPath = path.join(self.dbDir, self.resultsDirName)
        self.benchmarksFilePath = path.join(
            self.resultsDirPath, self.benchmarksFileName
        )

    ###########################################################################
    def __removeFiles(self, fileList):
        for f in fileList:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass

    def __isS3URL(self, url):
        """
        Returns True if url is a S3 URL, False otherwise.
        """
        if url.startswith("s3:"):
            return True

        return False

    def __waitForWrite(self):
        """
        Testing helper: pause for self.writeDelay seconds, or until
        self.cancelWrite turns True. Always set self.cancelWrite back to False
        so future writes can take place by default.

        Return True to indicate a write operation should take place, False to
        cancel the write operation, based on if the write was cancelled or not.
        """
        if not (self.cancelWrite):
            st = now = time.time()
            while ((now - st) < self.writeDelay) and not (self.cancelWrite):
                time.sleep(0.01)
                now = time.time()

        retVal = not (self.cancelWrite)
        self.cancelWrite = False
        return retVal

#!/cm/local/apps/python3/bin/python
#
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from __future__ import annotations

import concurrent.futures
import importlib.machinery
import importlib.util
import os
import re
import redfish
import socket
import sys
from datetime import datetime


from base_sampler import Sampler


def import_path(path, prefix="/cm/local/apps/cmd/scripts/metrics/configfiles/"):
    try:
        if prefix:
            path = prefix + path
        module_name = os.path.basename(path).replace("-", "_")
        spec = importlib.util.spec_from_loader(module_name, importlib.machinery.SourceFileLoader(module_name, path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[module_name] = module
        return module
    except (OSError, ImportError):
        return None


class RedfishDynamicSampler(Sampler):
    """
    Samples dynamic BMC monitoring data using the Redfish API
    """

    skipped = {"Name", "Id", "@odata.id", "@odata.type", "DataSourceUri"}
    postfixes_skip = (
        "@odata.id",
        "Type",
        "Number",
        "Serial",
        "Model",
        "UUID",
        "Manufacturer",
        "ErrorCorrection",
        "PortProtocol",
        "MaxLanes",
        "AllowedSpeedsMHz",
        "@odata.context",
        "@odata.etag",
    )

    def __init__(
        self,
        script: str,
        debug: bool = False,
        indent: int = 4,
        subdevice: str | None = None,
        environ: dict[str, str] | None = None
    ):
        super().__init__(script, debug, indent)

        # Field to store the Redfish client object
        self.rf = None
        self.hpe = False
        self.max_thread = 8
        self.merge_state = False
        self.subdevice = "" if subdevice is None else f"{subdevice}_"
        self.environ = environ

        # Health status from Redfish API is an enum of the following values
        self.performance_enum = ["Normal", "Boost", "Low"]
        self.health_enum = ["OK", "Warning", "Critical"]
        self.power_enum = ["On", "Off"]
        self.link_enum = [
            "NoStateChange",
            "Sleep",
            "Polling",
            "Disabled",
            "PortConfigurationTraining",
            "LinkUp",
            "LinkErrorRecovery",
            "PhyTest",
        ]
        # KDR: best guess replacement list (Redfish has differnet valies
        self.link_down_reason_code = [
            "NoLinkDown",
            "Unknown",
            "HighBitErrorRate",
            "BlockLockLost",
            "AlignmentLost",
            "FECSyncLost",
            "PllLockLost",
            "FIFOOverflow",
            "FalseSkipDetected",
            "MinorErrorThresholdExceeded",
            "PhyRetransmitTimeout",
            "HeartbeatErrors",
            "CreditMonitorWatchdogTimeout",
            "LinkLayerIntegrityThresholdExceeded",
            "LinkLayerBufferOverrun",
            "OOBCommandLinkHealthy",
            "OOBCommandLinkHighBER",
            "InbandCommandLinkHealthy",
            "InbandCommandLinkHighBER",
            "VerificationGatewayDown",
            "RemoteFaultReceived",
            "TrainingSequenceReceived",
            "ManagementCommandDown",
            "CableDisconnected",
            "CableAccessFault",
            "ThermalShutdown",
            "CurrentIssue",
            "PowerBudgetExceeded",
            "FastRawBERRecovery",
            "FastEffectiveBERRecovery",
            "FastSymbolBERRecovery",
            "FastCreditWatchdogRecovery",
            "PeerSleep",
            "PeerDisabled",
            "PeerDisableLocked",
            "PeerThermalEvent",
            "PeerForcedEvent",
            "PeerResetEvent",
        ]
        self.state_enum = ["Disabled", "Enabled"]

        # Base class for metrics in cmdaemon, will be extended with the name of the  metric collection
        self.base_class = "Environmental/Redfish/{}"

        # Mapping from metric to description, to be used when compiling the list of measurements
        self.metric_descriptions = {
            "chassis_health": "Health status of chassis",
            "cpu_health": "Health status of processor",
            "memory_health": "Health status of memory",
            "storage_health": "Health status of storage",
            "device_health": "Health status of storage device",
            "drive_health": "Health status of storage drive",
            "volume_health": "Health status of storage volume",
            "psu_health": "Health status of power supply",
            "psu_power": "The average power output in Watts of power supply",
            "fan_health": "Health status of a fan",
            "fan_speed": "Speed of a fan",
            "sensor_health": "Health status of sensor",
            "sensor_reading": "Reading of sensor",
            "pcie_health": "Health status of PCIe device",
            "manager_health": "Health status of manager (e.g., HPE iLO)",
        }

        # Metrics that are disabled (value `False`) will be skipped during sampling
        self.enabled = {
            "chassis_health": True,
            "cpu_health": True,
            "memory_health": True,
            "storage_health": True,
            "device_health": True,
            "drive_health": True,
            "volume_health": True,
            "psu_health": True,
            "psu_power": True,
            "fan_health": True,
            "fan_speed": True,
            "sensor_health": True,
            "sensor_reading": True,
            "pcie_health": True,
            "manager_health": True,
        }

        self.measurements = []
        self.metric_schemas = {}
        self.skipped_count = 0
        self.redfish_filter = None
        self.type_units = {
            "Power": "W",
            "Temperature": "C",
            "EnergyJoules": "J",
            "Current": "A",
            "Voltage": "V",
            "Frequency": "Hz",
            "EnergykWh": "kWh",
            "Percent": "%",
            "Rotational": "RPM",
            "Altitude": "Pa",
        }

        self.units = {
            "Watts": "W",
            "Celsius": "C",
            "Amps": "A",
        }
        self.fan_pattern = re.compile(r"FAN_(\d+)_(\w+)")
        self.cpu_pattern = re.compile(r"CPU_(\d+)")
        self.gpu_pattern = re.compile(r"GPU_(\d+)")
        self.core_util_pattern = re.compile(r"CoreUtil_(\d+)")
        self.processor_module_pattern = re.compile(r"ProcessorModule_(\d+)")
        self.fpga_pattern = re.compile(r"FPGA_(\d+)")
        self.mac_pattern = re.compile(r"([0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5})")
        self.nvlink_pattern = re.compile(r"NVLink_(\d+)")
        self.nvlink_management_pattern = re.compile(r"^(.*NVLinkManagement)_(\d+)_Port(.*)$")
        self.interswitch_pattern = re.compile(r"^(.*InterswitchPort)_(\d+)_Port(.*)$")
        self.iso8601_pattern = re.compile(r"P(?:(?P<days>\d+)D)?T?(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>[\d.]+)S)?")

        # Extensible device pattern matching - add new patterns here
        self.device_patterns = [
            re.compile(r"Power Supply (p?\d+)"),
            re.compile(r"power unit (\d+)", re.IGNORECASE),
            re.compile(r"Psu(\d+)", re.IGNORECASE),
        ]
        self.totals = {
            f"RF_{self.subdevice}PDB_TOTAL_PWR": re.compile(rf"^RF_{self.subdevice}PDB_\d_HSC_\d_[pP][wW][rR]_\d$"),
            f"RF_{self.subdevice}PDB_TOTAL_CURRENT": re.compile(rf"^RF_{self.subdevice}PDB_\d_HSC_\d_[cC][uU][rR]_\d$"),
            f"RF_{self.subdevice}PDB_TOTAL_VOLT_IN": re.compile(rf"^RF_{self.subdevice}PDB_\d_HSC_\d_Volt_In_\d$"),
            f"RF_{self.subdevice}PDB_TOTAL_VOLT_OUT": re.compile(rf"^RF_{self.subdevice}PDB_\d_HSC_\d_Volt_Out_\d$"),
            f"RF_{self.subdevice}NVLink_Port_TX_Total": re.compile(rf"^RF_{self.subdevice}NVLink_Port_TX$"),
            f"RF_{self.subdevice}NVLink_Port_RX_Total": re.compile(rf"^RF_{self.subdevice}NVLink_Port_TX$"),
            f"RF_{self.subdevice}NVLink_Port_RXErrors_Total": re.compile(rf"^RF_{self.subdevice}NVLink_Port_RXErrors$"),
            f"RF_{self.subdevice}NVLink_Port_TXErrors_Total": re.compile(rf"^RF_{self.subdevice}NVLink_Port_TXErrors$"),
        }
        self.averages = {
            f"RF_{self.subdevice}GPU_Processor_BandwidthUtilization": re.compile(rf"^RF_{self.subdevice}GPU_.*_Processor_BandwidthUtilization$"),
            f"RF_{self.subdevice}GPU_Memory_BandwidthUtilization": re.compile(rf"^RF_{self.subdevice}GPU_.*_Memory_BandwidthUtilization$"),
            f"RF_{self.subdevice}Average_Temp": re.compile(rf"^RF_{self.subdevice}.*_Temp(_\d)?$"),
        }

    def is_open(self, ip: str, port: int) -> bool:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect((ip, port))
            s.shutdown(2)
            return True
        except OSError as e:
            self.log(str(e))
            return False

    def load_env(self) -> bool:
        """
        Establish connection with Redfish server and create Redfish client object
        """
        # Retrieve credentials from environment variables
        sys.path.insert(0, "/cm/local/apps/cm-bios-tools/python")
        import cmd.env as cmd_env

        env_vars = cmd_env.CmdBmcEnvVars(self.environ)
        # check if IP is reachable first
        if not self.is_open(env_vars.remote_host, env_vars.port()):
            return False

        self.log("Opening a Redfish session")
        try:
            # Create a Redfish client object
            if bool(env_vars.username):
                self.rf = redfish.redfish_client(
                    base_url=env_vars.url(),
                    username=env_vars.username,
                    password=env_vars.password,
                )
            else:
                self.rf = redfish.redfish_client(base_url=env_vars.url())
            # Login with the Redfish client
            self.rf.login(auth="session")
        except redfish.rest.v1.RetriesExhaustedError:
            sys.stderr.write("Redfish retries exhausted\n")
            return False
        except redfish.rest.v1.SessionCreationError:
            sys.stderr.write("Redfish session failed\n")
            return False

        return True

    def initialize(self):
        """
        Initialize defines which data the producer will generate, executed:
        * when cmd starts
        * when a manual request from a client
        * when a new metric or check was provided in sample

        Note that we are returning the actual sample results already, because by
        nature of the monitoring we do not know what parts a system contains
        (e.g., how many fans) until we query the Redfish API. CMDaemon ignores
        the fields that are not relevant in this initialization stage, so there
        is no harm in returning *more* than necessary.
        """
        return self.sample()

    def unit_to_si(self, orig_value, orig_unit):
        prefixes = {"K": 10**3, "Mi": 1048576, "M": 10**6, "G": 10**9}
        units = {"By": "B", "Cel": "C"}
        new_unit = orig_unit
        new_value = orig_value
        for pfx, multiplier in prefixes.items():
            if orig_unit.startswith(pfx):
                new_value = orig_value * multiplier
                offset = len(pfx)
                new_unit = orig_unit[offset:]
                break
        new_unit = units.get(new_unit, orig_unit)
        return new_value, new_unit

    def _add_measure(
        self,
        metric_class: str,
        metric: str,
        value: str | float,
        unit: str = "",
        timestamp: int | str = 0,
        description: str = "",
        minimal: int = 0,
        maximal: int = 100,
        enum=None,
        url: str = ""
    ):
        """
        Helper method to construct measurements with all the necessary object fields
        and add them to the list that will ultimately be reported.
        """
        if (
            "@odata" in metric
            or (isinstance(value, str) and (not bool(value) or value.startswith("0x")))
            or isinstance(metric, list)
            or ("#" in metric and metric.endswith("target"))
            or "#Chassis.Reset" in metric
            or "BackgroundCopyStatus" in metric
            or "LinkState" in metric
            or "LocationContext" in metric
            or metric.endswith("Type_PCIeInterface")
            or metric.endswith("SKU")
            or metric.endswith("AssetTag")
            or metric.endswith("Model_ProcessorSummary")
        ):
            return

        # Base case: do not add this measurement if the metric was disabled
        if self.hpe and not self.enabled.get(metric, True):
            return
        if isinstance(value, list):
            self.log(f"List passed: {metric}")
            self.log(f"List passed: {value}")
            return
        if isinstance(value, dict):
            self.log(f"Dict passed: {metric}")
            self.log(f"Dict passed: {value}")
            return

        enum = [] if enum is None else enum

        orig_name = metric
        metric = metric.removesuffix("_Networking")
        metric = metric.replace("Port Metrics", "Port")
        metric = metric.replace("PortRX", "Port_RX")
        metric = metric.replace("PortTX", "Port_TX")
        metric = metric.replace("Metrics for", "")
        metric = metric.replace("Metrics", "")
        metric = metric.replace("___", "_")
        metric = metric.replace("__", "_")
        if isinstance(value, str):
            if value.isdigit():
                value = int(value)
            elif isinstance(value, bool):
                value = int(value)
            elif value.startswith("PT0"):  # ISO8601 duration
                multiplier = 1
                if value.endswith("M"):
                    multiplier = 60
                elif value.endswith("H"):
                    multiplier = 3600
                elif value.endswith("D"):
                    multiplier = 86400
                value = float(value.strip("PT")[:-1]) * multiplier
                unit = "s"
            elif value.startswith("{") and value.endswith("}"):
                return
            elif match := self.iso8601_pattern.match(value):
                days = int(match.group("days") or 0)
                hours = int(match.group("hours") or 0)
                minutes = int(match.group("minutes") or 0)
                seconds = float(match.group("seconds") or 0)
                value = days * 86400 + hours * 3600 + minutes * 60 + seconds
                unit = "s"

        if metric_class in self.metric_descriptions:
            metric_class = metric_class.split("_")[1]

        cumulative = False
        parameter = []

        if res := self.cpu_pattern.search(metric):
            cpu = res.group(1)
            if res := self.processor_module_pattern.search(metric):
                parameter.append(f"module={res.group(1)}")
                metric = metric.replace(f"Module_{res.group(1)}", "")
            parameter.append(f"cpu={cpu}")
            metric = metric.replace(f"_CPU_{cpu}", "")
            if res := self.core_util_pattern.search(metric):
                parameter.append(f"core={res.group(1)}")
                metric = metric.replace(f"CoreUtil_{res.group(1)}", "Util")
        elif res := self.fan_pattern.search(metric):
            side = res.group(2).lower()
            metric = metric.replace(f"_FAN_{res.group(1)}_{res.group(2)}", "")
            if side == "pwm":
                metric += "_fanpwm"
                parameter.append(f"fan={res.group(1)}")
            else:
                metric += "_fanspeed"
                parameter.append(f"fan={res.group(1)};side={side}")
        elif res := self.fpga_pattern.search(metric):
            parameter.append(f"fpga={res.group(1)}")
            metric = metric.replace(f"_FPGA_{res.group(1)}", "")

        mac_num = self.mac_pattern.findall(metric)
        if mac_num and metric.endswith(mac_num[0]):
            metric = metric.replace(mac_num[0], "").replace("-", "").rstrip()
            parameter.append(mac_num[0].replace(":", "-"))

        # KDR: no Count -> cumulative
        if metric.endswith("MHz") or metric.endswith("Mhz"):
            unit = "Hz"
            value = value * 10**6
            metric = metric.removesuffix("MHz")
            metric = metric.removesuffix("Mhz")
            if description:
                description = description.replace("MHz", "Hz")
                description = description.replace("Mhz", "Hz")
        elif metric.endswith("GiB_MemorySummary"):
            unit = "B"
            value = value * 2**30
            metric = metric.removesuffix("GiB_MemorySummary")
        elif metric.endswith("Gbps"):
            unit = "b/s"
            value = value * 10**9
            metric = metric.removesuffix("Gbps")
            if description:
                description = description.replace("Gbps", "b/s")
        elif metric.endswith("MiB"):
            metric = metric.removesuffix("MiB")
            if description:
                description = description.replace("MiB", "B")
        elif metric.endswith("kWh"):
            metric = metric.replace("kWh", "")
            unit = "Ws"
            value = float(value) * 1000 * 3600
        elif metric.endswith("FanSpeedPercent") and float(value) > 100.0:
            metric = metric.replace("SpeedRPM FanSpeedPercent", "FanSpeed")
            unit = "RPM"
        elif metric.endswith("FanSpeed"):
            metric = metric.removesuffix("SpeedRPM")
            unit = "RPM"
        elif metric.endswith("BandwidthPercent"):
            metric = metric.replace("BandwidthPercent", "BandwidthUtilization")
            unit = "%"
        elif metric.endswith("UtilizationPercent"):
            metric = metric.removesuffix("Percent")
            unit = "%"
        elif metric.endswith("Bytes"):
            metric = metric.removesuffix("Bytes")
            unit = "B"
            cumulative = True
        elif metric.endswith("Time"):
            metric = metric.removesuffix("Time")
            unit = "s"
        else:
            for k, v in self.type_units.items():
                if metric.endswith(k):
                    unit = v
                    break
            if "PowerFactor" in metric:
                metric = metric.replace("PowerWatts", "")
            else:
                for k, v in self.units.items():
                    if metric.endswith(k):
                        metric = metric.replace(k, "")
                        unit = v
                        break

        if res := self.gpu_pattern.search(url):
            parameter.append(f"gpu={res.group(1)}")
            if metric.startswith("GPU"):
                metric = metric.replace(f"GPU_{res.group(1)}", "")
            else:
                metric = metric.replace(f"_GPU_{res.group(1)}", "")

        if res := self.nvlink_pattern.search(metric):
            parameter.append(f"port={res.group(1)}")
            metric = metric.replace(f"_{res.group(1)}", "")
            if metric.endswith("State Status"):
                enum = self.state_enum
        elif res := self.nvlink_management_pattern.search(metric):
            parameter.append(res.group(2))
            metric = res.group(1) + res.group(3)
            if metric.endswith("State Status"):
                enum = self.state_enum
        elif res := self.interswitch_pattern.search(metric):
            parameter.append(res.group(2))
            metric = res.group(1) + res.group(3)
            if metric.endswith("State Status"):
                enum = self.state_enum
        else:
            # Check extensible device patterns (PSU, power unit, etc.)
            for i, pattern in enumerate(self.device_patterns):
                if match := pattern.search(metric):
                    # For the first pattern, only remove the number (group 1)
                    # For others, remove the entire matched pattern
                    if i == 0:
                        metric = metric.replace(match.group(1), "")
                        # Extract only numeric part for parameter
                        param_value = ''.join(filter(str.isdigit, match.group(1)))
                        parameter.append(str(int(param_value) + 1))
                    else:
                        metric = metric.replace(match.group(0), "")
                        parameter.append(match.group(match.lastindex))
                    break

        if extra_description := self.metric_descriptions.get(metric_class, None):
            if bool(description):
                description = extra_description + ": " + description
            else:
                description = extra_description

        if not self.hpe and metric_class in self.metric_descriptions:
            metric_class = metric_class.split("_")[0]
        metric_class = self.base_class.format(metric_class.replace("_", " ").replace("Metrics", "")).rstrip("/")
        if "_CPU_" in orig_name and not metric_class.endswith("/CPU"):
            metric_class += "/CPU"

        if isinstance(value, bool):
            metric = metric.removesuffix("Enabled").removesuffix("Enable")
            value = self.state_enum[int(value)]
            enum = self.state_enum
        elif any("ealth" in word for word in (metric, parameter, metric_class)):
            enum = self.health_enum
            parameter.append("state=yes")

        metric = metric.replace(" ", "_")
        metric = f"RF_{self.subdevice}{metric}".replace("__", "_")
        description = description.rstrip(".") if bool(description) else ""
        parameter = ";".join(parameter)
        m = {
            "metric": metric,
            "parameter": parameter,
            "bcm": f"{metric}:{parameter}",
            "class": metric_class,
            "description": description,
            "value": value,
            "cumulative": cumulative,
            "details": {
                "url": url,
                "field": orig_name,
            },
        }

        if unit:
            m["unit"] = unit
            if unit == "%":
                m["min"] = 0.0
                m["max"] = 1.0
                m["value"] = value / maximal
            elif isinstance(m["value"], int | float):
                m["value"], m["unit"] = self.unit_to_si(value, unit)
                if unit in m.get("description", ""):
                    m["description"] = m["description"].replace(unit, m["unit"])

        if enum:
            m["enum"] = enum
        elif metric.endswith("PowerState"):
            m["enum"] = self.power_enum
        elif metric.endswith("LinkStatus"):
            m["enum"] = self.link_enum
        elif metric.endswith("LinkDownReasonCode"):
            m["enum"] = self.link_down_reason_code
        elif metric.endswith("PowerBreakPerformanceState"):
            m["enum"] = self.performance_enum
        elif metric.endswith("State") or metric.endswith("State_Status"):
            m["enum"] = self.state_enum
        elif m["value"] in self.health_enum:
            m["enum"] = self.health_enum
        elif m["value"] in self.state_enum:
            m["enum"] = self.state_enum
        elif m["value"] in self.performance_enum:
            m["enum"] = self.performance_enum

        if timestamp:
            if isinstance(timestamp, str):
                m["time"] = int(datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S%z").timestamp() * 1000)
            else:
                m["time"] = timestamp

        self.measurements.append(m)

    def gather_chassis_sensors(self):
        return set(
            sensor["@odata.id"]
            for chassis in self.get_members(self.rf.root, "Chassis")
            for sensor in sensors
        )

    def sample_hpe(self):
        """
        Sample produces the actual data.
        Execution depends on the settings:
        * periodic every X seconds
        * on demand
        * during prejob
        """

        chassis = self.get_members(self.rf.root, "Chassis")
        self.add_health_metrics("chassis_health", chassis)

        [
            (
                self.monitor_health(system, "Processors", "cpu_health"),
                self.monitor_health(system, "Memory", "memory_health"),
                self.monitor_simplestorage(system),
                self.monitor_storage(system),
            )
            for system in self.get_members(self.rf.root, "Systems")
        ]

        [
            (
                self.monitor_powersupplies(c),
                self.monitor_fans(c),
                self.monitor_temperatures(c),
                self.monitor_pcie(c),
            )
            for c in chassis
        ]

        self.monitor_manager(self.rf.root)
        return self.measurements

    def get_metric_schema(self, metric_type):
        return self.metric_schemas.get(metric_type, None)

    def get_metric_property(self, kind, metric, prop, subtype, immediate=False):
        if not self.get_metric_schema(kind):
            self.log(f"Can not get metric schema for {kind}")
            return None
        metric_def = self.get_metric_schema(kind)["definitions"].get(subtype, None)
        if not metric_def:
            self.log(f"Getting {kind} from definitions")
            metric_def = self.get_metric_schema(kind)["definitions"].get(kind, None)
        if not metric_def:
            self.log("Failed to get definition for {subtype}/{kind} in schema {kind}")
            return None
        if immediate:
            self.log(f"Getting {prop} from metric")
            return metric_def.get(prop, None)

        result = None
        key = metric_def["properties"].get(metric, None)
        if key:
            result = key.get(prop, None)
            if not result and prop == "enum":
                self.log(f"Failed to find property {prop} in metric {metric} ")
                ref = key.get("$ref", None)
                if ref and ref.startswith("#"):
                    self.log(f"Reference to the same schema: {ref}")
                    result = self.get_metric_property(kind, metric, prop, metric, immediate=True)
                    self.log(f"result = {result}")
        else:
            self.log(f"Failed to find property {metric} in schema {kind}/{subtype} ")
            stype = metric_def["properties"].get(subtype, None)
            if stype:
                ref = stype.get("$ref", None)
                if ref:
                    self.log(f"Reference address: {ref}")
                    ref_kind = ref.split("/")[5].split(".")[0]
                    self.log(f"Reference kind: {ref_kind}")
                    result = self.get_metric_property(ref_kind, metric, prop, subtype)
                else:
                    result = stype.get(prop, None)
                    if result:
                        self.log(f"Failed back to parent property: {result}")
        return result

    def filter_metric(self, name):
        if self.redfish_filter:
            return (
                name in self.redfish_filter.exact or
                any(regex.findall(name) for regex in self.redfish_filter.regex)
            )
        return False

    def skip_metric(self, name):
        if name in self.skipped or any(name.endswith(x) for x in self.postfixes_skip):
            self.log(f"Skipping {name}")
            if "@odata.id" not in name:
                self.skipped_count += 1
            return True
        return False

    def add_metric(self, base, name, metric_type, value, url="", unit=""):
        description = self.get_metric_property(
            metric_type,
            name,
            "description",
            metric_type,
        )
        if unit == "":
            if name == "CoreVoltage":
                unit = "V"
            else:
                unit = self.get_metric_property(metric_type, name, "units", metric_type)
        enum = self.get_metric_property(metric_type, name, "enum", metric_type)
        self.log(f"{name}: Enum for {metric_type} is {enum}")
        self._add_measure(
            metric_type,
            base + " " + name,
            value,
            description=description,
            unit=unit,
            enum=enum,
            url=url,
        )

    def add_metric_reading_type(self, base, metric, url="", key_mapping=None):
        metric_name = metric.get("Name")
        metric_type = metric.get("ReadingType")
        metric_value = metric.get("Reading")
        if metric_type == "Rotational" and not key_mapping:
            metric_name = metric_name.replace("fan1", "fan speed")
        metric_unit = metric.get("ReadingUnits")
        if metric_unit == "Celsius":
            metric_unit = "C"

        # Normalize metric_name: glue together Input/Output + Current/Power/Voltage
        if metric_name:
            words = metric_name.split()
            if len(words) >= 2:
                # Look for consecutive pairs of input/output + current/power/voltage
                for i in range(len(words) - 1):
                    first_word_lower = words[i].lower()
                    second_word_lower = words[i + 1].lower()
                    if first_word_lower in ('input', 'output') and second_word_lower in ('current', 'power', 'voltage'):
                        # Glue these two words together, converting output to Rail
                        first_part = "Rail" if first_word_lower == "output" else words[i].capitalize()
                        glued = first_part + words[i + 1].capitalize()
                        words[i] = glued
                        words.pop(i + 1)
                        break
                metric_name = ' '.join(words)

        # Apply key_mapping if provided
        if metric_name and key_mapping:
            # Try exact match first
            if metric_name in key_mapping:
                metric_name = key_mapping[metric_name]
            else:
                # Try substring match - replace matched part but keep the rest
                metric_name_lower = metric_name.lower()
                for key, value in key_mapping.items():
                    if key in metric_name_lower:
                        start_idx = metric_name_lower.index(key)
                        metric_name = metric_name[:start_idx] + value + metric_name[start_idx + len(key):]
                        break


        if metric_name:
            self.add_metric(base, metric_name, metric_type, metric_value, url, unit=metric_unit)
        if "RailPower" in metric_name and "chassis" not in metric_name.lower():
            self.add_metric(base, metric_name.replace("RailPower", "OutputPower"), metric_type, metric_value, url, unit=metric_unit)

    def process_list(self, base, name, metric_type, lst, url=""):
        if bool(lst):
            value = lst[0].get("Reading", None)
            if not bool(value):
                value = " ".join(map(str, lst))
            self.add_metric(base, name, metric_type, value, url)

    def add_metric_from_dict(self, dic, base, name, metric_type, url=""):
        for key, value in dic.items():
            if self.skip_metric(key):
                continue
            description = self.get_metric_property(metric_type, key, "description", name)
            unit = ""
            if not key.endswith("Count"):
                unit = self.get_metric_property(
                    metric_type,
                    key,
                    "units",
                    name,
                )
            enum = self.get_metric_property(metric_type, name, "enum", metric_type)
            self.log(enum)
            self._add_measure(
                metric_type,
                base + "_" + name + "_" + key,
                value,
                description=description,
                unit=unit,
                enum=enum,
                url=url,
            )

    def process_dict(self, base, m, metric_type, dct, url=""):
        if len(dct) == 1:
            for k, v in dct.items():
                if isinstance(v, dict):
                    self.log(f"Dict inside a dict detected: {dct}")
                    self.add_metric_from_dict(v, base, k, metric_type, url)
                    return
        if "Value" in dct and "Unit" in dct:
            unit = dct["Unit"]
            if unit == "\u00b0C":
                unit = "C"
            self.add_metric(base, m, metric_type, dct["Value"], url, unit=unit)
            return
        self.log(f"Value is compound: {m}")
        for k, v in dct.items():
            if k == "Reading":
                self.add_metric(base, m, metric_type, v, url)
            elif k != "DataSourceUri":
                self.add_metric(base + k, m, metric_type, v, url)

    def sample(self):
        manufacturer = ""
        model = ""
        uris = ["/redfish/v1/Chassis/DGX", "/redfish/v1/Chassis/BMC_0"]
        for uri in uris:
            doc = self.get(uri)
            manufacturer = doc.obj.get("Manufacturer", None)
            model = doc.obj.get("Model", None)
            if manufacturer and model:
                break
        config = None
        if manufacturer == "NVIDIA":
            if model == "DGXH100":
                config = import_path("redfish-h100.conf")
                self.log("Loaded H100 config\n")
            elif model == "P4265":
                config = import_path("redfish-gh200.conf")
                self.log("Loaded GH200 config\n")
            elif model == "GB200 NVL":
                config = import_path("redfish-gb200.conf")
                self.log("Loaded GB200 config\n")
            elif model == "GB300 NVL":
                config = import_path("redfish-gb300.conf")
                self.log("Loaded GB300 config\n")
            elif model == "DGXB300":
                config = import_path("redfish-b300.conf")
                self.log("Loaded B300 config\n")
            else:
                config = import_path("redfish-generic.conf")
                self.log(f"Unknown model: {model}")
        else:
            config = import_path("redfish-generic.conf")
            self.log(f"Generic config for: {manufacturer}")
        if config:
            self.redfish_filter = config.RedfishFilter()
        else:
            self.log("Error loading config")
            return []

        if self.redfish_filter.get_schemas:
            self.metric_schemas = {it["Id"]: it for it in self.get_members(None, "JsonSchemas", self.get_schema)}

        self.redfish_filter.regex = [re.compile(it) for it in self.redfish_filter.regex]

        if hasattr(self.redfish_filter, "max_thread"):
            self.max_thread = self.redfish_filter.max_thread
        if hasattr(self.redfish_filter, "merge_state"):
            self.merge_state = self.redfish_filter.merge_state

        if bool(self.redfish_filter.endpoints):
            metric_endpt = self.redfish_filter.endpoints
        elif telemetry_service := self.rf.root.get("TelemetryService", None):
            telemetry = self.get(telemetry_service["@odata.id"])
            metric_definitions = self.get_members(telemetry.obj, "MetricDefinitions")
            metric_desc = {
                md["Id"]: {
                    "units": md.get("Units", ""),
                    "description": md.get("Description", "").replace("Metric definition for ", ""),
                    "min": md.get("MinReadingRange", 0),
                    "max": md.get("MaxReadingRange", 100),
                }
                for md in metric_definitions
            }
            metric_values = sum(
                [mr.get("MetricValues", []) for mr in self.get_members(telemetry.obj, "MetricReports")], []
            )
            metric_endpt = [
                it["MetricProperty"].split("#")[0] for it in metric_values if "MetricProperty" in it
            ]
            [
                self._add_measure(
                    "sensor_reading",
                    metric["MetricId"],
                    metric["MetricValue"],
                    metric_desc.get(metric["MetricId"], {}).get("units", ""),
                    metric.get("Timestamp", 0),
                    metric_desc.get(metric["MetricId"], {}).get("description", ""),
                    metric_desc.get(metric["MetricId"], {}).get("min", ""),
                    metric_desc.get(metric["MetricId"], {}).get("max", ""),
                )
                for metric in metric_values
                if "MetricProperty" not in metric
            ]
            #if len(metric_endpt) != len(metric_values):
                #self.hpe = True
                #metric_endpt.extend(self.gather_chassis_sensors())
        else:
            metric_endpt = []

        metric_endpt = [it for it in metric_endpt if not self.filter_metric(it)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_thread) as executor:
            futures = {executor.submit(self.get, url): url for url in metric_endpt}
            for future in concurrent.futures.as_completed(futures.keys()):
                url = futures[future]
                try:
                    metric = future.result().dict
                    mtype = metric.get("ReadingType", None)
                    if not mtype:
                        if otype := metric.get("@odata.type", None):
                            if metric_name := metric.get("Name", None):
                                metric_type = otype.split(".")[-1]
                                metric = {m: mvalue for m, mvalue in metric.items() if not self.skip_metric(m)}
                                [
                                    self.process_dict(metric_name, m, metric_type, mvalue, url)
                                    for m, mvalue in metric.items()
                                    if isinstance(mvalue, dict)
                                ]
                                [
                                    self.process_list(metric_name, m, metric_type, mvalue, url)
                                    for m, mvalue in metric.items()
                                    if isinstance(mvalue, list)
                                ]
                                [
                                    self.add_metric(metric_name, m, metric_type, mvalue, url)
                                    for m, mvalue in metric.items()
                                    if not isinstance(mvalue, (list, dict))
                                ]
                    if status := metric.get("Status", None):
                        if metric_id := metric.get("Id", None):
                            if status.get("State", None) == "Enabled":
                                if metric_name := metric.get("Name", None):
                                    self.log(metric)
                                    self._add_measure(
                                        "sensor_health",
                                        metric_id,
                                        status.get("Health", ""),
                                        description=metric_name,
                                        url=url,
                                    )
                                    if bool(mtype):
                                        if unit := self.type_units.get(mtype, metric.get("ReadingUnit", None)):
                                            self._add_measure(
                                                "",
                                                metric_id,
                                                metric.get("Reading", None),
                                                unit=unit,
                                                description=metric_name,
                                                url=url,
                                            )
                            else:
                                self.log(f"Skipping: {mtype}, not enabled")
                        else:
                            self.log(f"Skipping: {mtype}, no id")
                    else:
                        self.log(f"Skipping: {mtype}, no status")

                except Exception as exc:
                    import traceback

                    self.log(f"{future!r} generated an exception: {exc}")
                    self.log(traceback.format_exc())
        if self.hpe:
            self.sample_hpe()

        if self.merge_state:
            states = {
                it["metric"]: it
                for it in self.measurements
                if it.get("parameter", None) == "state"
            }
            for it in self.measurements:
                if state := states.get(it["metric"], None):
                    if state != it:
                        it["info"] = state["value"]
                        state["drop"] = True
            self.measurements = [it for it in self.measurements if not it.get("drop", False)]

        for k, v in self.totals.items():
            matches = [x for x in self.measurements if isinstance(x["value"], (int, float)) and v.fullmatch(x["metric"])]
            if bool(matches):
                total = {
                    "metric": k,
                    "value": sum(x["value"] for x in matches),
                    "class": matches[0].get("class", ""),
                    "unit": matches[0].get("unit", "B"),
                }
                self.measurements.append(total)

        for k, v in self.averages.items():
            matches = [x for x in self.measurements if isinstance(x["value"], (int, float)) and v.fullmatch(x["metric"])]
            if bool(matches):
                value = sum(x["value"] for x in matches) / len(matches)
                total = {
                    "metric": k,
                    "value": value,
                    "class": matches[0].get("class", ""),
                    "unit": matches[0].get("unit", "B"),
                }
                self.measurements.append(total)

        self.log(f"Total: {len(self.measurements)}")
        self.log(f"Skipped: {self.skipped_count}")
        return self.measurements

    def get_members(self, root, collection_name: str, call=None):
        """
        Helper method to fetch all Members of a collection.
        """
        if root is None:
            root = self.rf.root
        if call is None:
            call = self.get

        try:
            uri = root[collection_name]
            if not isinstance(uri, dict):
                return []
            uri = uri["@odata.id"]
            res = self.get(uri)
            if not res or not res.obj["Members"]:
                return []
            members_uris = [m["@odata.id"] for m in iter(res.obj["Members"])]

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_thread) as executor:
                futures = [executor.submit(call, uri) for uri in members_uris]
                return [future.result().dict for future in futures]

        except KeyError:
            self.log(f'ERROR: Collection "{collection_name}" was not found')

        return []

    def add_health_metrics(self, metric: str, members: list):
        """
        Helper method to automatically add health measurement metrics for a list of members.
        """
        [
            self._add_measure("sensor_health", member["Id"], member["Status"]["Health"], enum=self.health_enum)
            for member in members
            if member["Status"]["State"] == "Enabled"
        ]

    def monitor_health(self, root, collection: str, metric: str):
        """
        Monitor health measurements for all members of a certain Collection
        """
        self.add_health_metrics(metric, self.get_members(root, collection))

    def monitor_simplestorage(self, system):
        """
        Monitor health measurements for all Devices in all SimpleStorage controllers in a System.
        """
        [
            self._add_measure("device_health", d["Id"], d["Status"]["Health"])
            for member in self.get_members(system, "SimpleStorage")
            for device in member["Devices"]
            if device["Status"]["State"] == "Enabled"
        ]

    def monitor_storage(self, system):
        """
        Monitor health measurements for all Drives in all Storage controllers in a System.
        """
        [
            (
                self.monitor_health(it, "Drives", "drive_health"),
                self.monitor_health(it, "Volumes", "volume_health"),
            )
            for t in self.get_members(system, "Storage")
        ]

    def _key_safe_access(self, root, collection_name, object_name):
        """
        Helper method to access an object in a collection safely; i.e., by catching
        any KeyError that might occur and logging it, but silently continuing with a returned
        empty list so that the metric will just not appear but the monitoring finishes.
        """
        try:
            uri = root[collection_name]["@odata.id"]
            res = self.get(uri)
            return res.obj.get(object_name, [])
        except KeyError:
            self.log(f'ERROR: Collection "{collection_name}" or object "{object_name}" was not found')
            return []

    def monitor_powersupplies(self, chassis):
        """
        Monitor health measurements for PowerSupplies in all Chassis.
        """
        [
            (
                self._add_measure("psu_health", psu["MemberId"], psu["Status"]["Health"]),
                self._add_measure("psu_power", psu["MemberId"], psu["LastPowerOutputWatts"], unit="W"),
            )
            for psu in self._key_safe_access(chassis, "Power", "PowerSupplies")
            if psu["Status"]["State"] == "Enabled"
        ]

    def monitor_fans(self, chassis):
        """
        Monitor health measurements for fans in all chassis, in addition to speed in %.
        """
        [
            (
                self._add_measure("fan_health", fan["Name"], fan["Status"]["Health"]),
                self._add_measure("fan_speed", fan["Name"], fan["Reading"] / 100, unit="%"),
            )
            for fan in self._key_safe_access(chassis, "Thermal", "Fans")
            if fan["Status"]["State"] == "Enabled"
        ]

    def monitor_temperatures(self, chassis):
        """
        Monitor health measurements for temperature sensors, in addition to their readings in Celsius.
        """
        [
            (
                self._add_measure("sensor_health", temp["Name"], temp["Status"]["Health"]),
                self._add_measure("sensor_reading", temp["Name"], temp.get("ReadingCelsius", None), unit="C"),
            )
            for temp in self._key_safe_access(chassis, "Thermal", "Temperatures")
            if temp["Status"]["State"] == "Enabled"
        ]

    def monitor_pcie(self, root):
        """
        Monitor health measurements for PCIe Devices
        """
        [
            self._add_measure("pcie_health", member["Name"], member["Status"]["Health"])
            for member in self.get_members(root, "PCIeDevices")
            if member["Status"]["State"] == "Enabled"
        ]

    def monitor_manager(self, root):
        """
        Monitor health measurements for BMC manager (e.g., HPE iLO).
        """
        try:
            if hpe := root["Oem"].get("Hpe", None):
                [
                    self._add_measure("manager_health", m["ManagerType"], m["Status"]["Health"])
                    for m in hpe["Manager"]
                ]
        except KeyError:
            self.log('ERROR: "Oem" collection was not found')

    def get_schema(self, uri: self):
        return self.get(self.get(uri).obj["Location"][0]["Uri"]).obj

    def get(self, uri: str):
        return self.rf.get(uri)

    def post(self, data: list):
        self.log("Ending the Redfish session")
        try:
            self.rf.logout()
        except Exception as e:
            self.log(f"Failed to end the Redfish session: {e}")


def main():
    interfaces = [name for name in os.environ.get("CMD_INTERFACES", "").split(" ") if name.startswith("rf")]
    if len(interfaces) <= 1:
        sampler = RedfishDynamicSampler(__file__)
        return sampler.run()
    # split BlueField RedFish into a new thread
    samplers = [RedfishDynamicSampler(__file__)]  # host
    bluefield = re.compile(r"^rf1(\d)$")          # BlueFields is always rf1<index>
    for name in interfaces:
        if match := bluefield.match(name):
            if ip := os.environ.get(f"CMD_INTERFACE_{name}_IP", None):
                environ = os.environ.copy()
                environ["CMD_BMCIP"] = ip
                if username := os.environ.get(f"CMD_INTERFACE_{name}_USERNAME", None):
                    environ["CMD_BMCUSERNAME"] = username
                if password := os.environ.get(f"CMD_INTERFACE_{name}_PASSWORD", None):
                    environ["CMD_BMCPASSWORD"] = password
                samplers = [RedfishDynamicSampler(__file__, subdevice=f"BF{match[1]}", environ=environ)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(samplers)) as executor:
        futures = [executor.submit(sampler.run) for sampler in samplers]
        return max(future.result() for future in futures)


if __name__ == "__main__":
    sys.exit(main())

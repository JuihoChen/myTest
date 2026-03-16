#!/bin/bash
# Author : NVIDIA

image_name=gb200-image

cmsh -c "softwareimage; clone default-image ${image_name}; commit"
cmsh -c "watch -n 3 task list"

################################################################################################

cm-chroot-sw-img /cm/images/${image_name}

## Add nvidia dgx repositories
## https://docs.nvidia.com/dgx/dgx-os-7-user-guide/installing_on_ubuntu.html#installing-dgx-system-configurations-and-tools
curl https://repo.download.nvidia.com/baseos/ubuntu/noble/arm64/dgx-repo-files.tgz | tar xzf - -C /
mv /etc/apt/sources.list.d/cm-cuda-ubuntu2404-sbsa.list /etc/apt/sources.list.d/cm-cuda-ubuntu2404-sbsa.list.disabled
#Disable dgx packages
cat > /etc/apt/preferences.d/black.pref << 'EOF'
Package: dgx*
Pin: release *
Pin-Priority: -1
EOF

#rm -f /dev/null; mknod -m 666 /dev/null c 1 3
apt update
kernel_version="6.8.0-1025-nvidia-64k"
apt-get install -y \
  linux-headers-${kernel_version} \
  linux-image-${kernel_version} \
  linux-modules-${kernel_version} \
  linux-tools-${kernel_version}
dkms status

apt-get install -y wget numactl ipmitool lldpd needrestart

## DGX packages: apt depends nvidia-system-core
apt install -y \
  cuda-compute-repo hpc-sdk-repo nv-grubmenu nv-grubserial ipmitool msecli nv-common-apis nv-cpu-governor nv-env-paths nv-iommu nv-ipmi-devintf nv-limits nv-update-disable nvgpu-services-list nvidia-acs-disable nvidia-disable-init-on-alloc nvidia-disable-numa-balancing nvidia-earlycon nvidia-enable-power-meter-cap nvidia-esm-hook-epilogue nvidia-fs-loader nvidia-kernel-defaults nvidia-nvme-options nvidia-pci-bridge-power nvidia-pci-realloc nvidia-raid-config nvidia-relaxed-ordering-gpu nvidia-redfish-config nvidia-relaxed-ordering-nvme nvme-cli nvidia-repo-keys nvidia-ipmisol tpm2-tools #dgx-release dgx-repo nvidia-crashdump nvidia-mig-manager

## DGX packages: apt depends nvidia-system-utils
apt install -y \
  nv-persistence-mode nvidia-modprobe nvidia-container-toolkit nvidia-fs-loader nvidia-logrotate nvidia-conf-cachefilesd #nvsm nvidia-motd
systemctl disable nvsm.service

## DGX packages: apt depends nvidia-system-extra
apt install -y \
  automake bash-completion bison build-essential chrpath cifs-utils cmake cryptsetup cryptsetup-initramfs curl docker-ce ethtool flex fping gdb gdisk git htop iperf libelf-dev libltdl-dev lsof lsscsi m4 mdadm net-tools nfs-common nv-docker-options openssh-server parted pciutils perftest pm-utils powercap-utils quota rasdaemon samba-common samba-libs sg3-utils shim-signed smartmontools sosreport ssh ssh-import-id swig sysstat udev vim vlan witalian wngerman wogerman wportuguese wspanish wswiss xserver-xorg #command-not-found

## GPU Driver: !!! It's important to make sure the all the nvidia.*570 packages' version the same !!!
dpkg -l|grep 'nvidia.*570' 
# As GPU driver packages already been installed in base image, if all the packages align each other, then no action needed.
# If not, uninstall the current ones
apt purge $(dpkg -l|grep 'nvidia.*570'|awk '{print $2}')
# Then install GPU driver from nvidia repo ONLY!!! The better to set the version explicitly.
apt update
apt-cache madison nvidia-imex-570 libnvidia-nscq-570 nvidia-driver-570-open | grep developer.download.nvidia.com
#ver=570.172.08
ver=570.133.20
apt install -y --allow-downgrades \
  nvidia-imex-570=${ver}-1 \
  libnvidia-nscq-570=${ver}-1  \
  nvidia-fabricmanager-570=${ver}-1  \
  nvidia-driver-570-open=${ver}-0ubuntu1 \
  nvidia-dkms-570-open=${ver}-0ubuntu1 \
  nvidia-settings=${ver}-0ubuntu1 \
  libxnvctrl0=${ver}-0ubuntu1 \
  nvidia-modprobe=${ver}-0ubuntu1 \
  nvidia-kernel-source-570-open=${ver}-0ubuntu1 \
  nvidia-kernel-common-570=${ver}-0ubuntu1 \
  libnvidia-gl-570=${ver}-0ubuntu1 \
  libnvidia-compute-570=${ver}-0ubuntu1 \
  libnvidia-extra-570=${ver}-0ubuntu1 \
  nvidia-compute-utils-570=${ver}-0ubuntu1 \
  libnvidia-decode-570=${ver}-0ubuntu1 \
  libnvidia-encode-570=${ver}-0ubuntu1 \
  nvidia-utils-570=${ver}-0ubuntu1 \
  xserver-xorg-video-nvidia-570=${ver}-0ubuntu1 \
  libnvidia-cfg1-570=${ver}-0ubuntu1 \
  libnvidia-fbc1-570=${ver}-0ubuntu1 \
  libnvidia-common-570=${ver}-0ubuntu1

apt install -y \
  libnvidia-container-tools libnvidia-container1 nvidia-container-toolkit nvidia-container-toolkit-base
apt install -y \
  datacenter-gpu-manager-4-core datacenter-gpu-manager-4-cuda12
systemctl enable nvidia-persistenced nvidia-dcgm nvidia-imex #nvidia-fabricmanager 
mkdir -p /var/run/nvidia-fabricmanager
chmod 755 /var/run/nvidia-fabricmanager

## CUDA
apt install -y cuda-nvml-dev-12-8 #cuda-toolkit-12-8

## GDRCOPY #packages from dgx
apt install -y gdrdrv-dkms gdrcopy

## DOCA #packages from dgx, nvidia-system-mlnx-drivers will install doca relate packages from DOCA repo
# apt depends nvidia-system-mlnx-drivers
apt install -y \
  doca-ofed mlnx-pxe-setup nvidia-mlnx-config nvidia-ib-umad-loader mlnx-nvme-dkms mlnx-nfsrdma-dkms mlnx-fw-updater
systemctl disable srp_daemon.service  srptools.service
systemctl enable openibd

## nvidia-peermem
apt install -y nvidia-peermem-loader

## For Lustre
cat >> /etc/sysctl.conf <<- EOF
### Add for DDN/Lustre
net.ipv4.conf.all.accept_local=1
net.ipv4.conf.all.arp_announce=2
net.ipv4.conf.all.arp_filter=0
net.ipv4.conf.all.arp_ignore=1
net.ipv4.conf.all.rp_filter=0
net.ipv4.conf.default.accept_local=1
net.ipv4.conf.default.arp_announce=2
net.ipv4.conf.default.arp_filter=0
net.ipv4.conf.default.arp_ignore=1
net.ipv4.conf.default.rp_filter=0
### Enabel Magic SysRq
kernel.sysrq=1
EOF

systemctl set-default multi-user.target
echo 'root:123456'|chpasswd
rm -rf /var/lib/command-not-found /var/cache/swcatalog
apt purge -y command-not-found appstream update-notifier-common
dpkg -l|grep -E 'command-not-found|appstream|update-notifier-common'

apt purge -y unattended-upgrades

sed -i.ori -e 's#next unless "$tag" ne "";#next unless defined $tag and "$tag" ne "";#g' /usr/bin/dshbak

#Fix pyxis issue on aarch64
apt download pyxis-sources
dpkg -i ./$(ls -1 pyxis-sources_*.deb)
rm -f pyxis-sources_*.deb
find /cm/local/apps/slurm -name spank.h
CFLAGS=-I/cm/local/apps/slurm/current/include /cm/local/apps/slurm/current/scripts/install-pyxis.sh
ls -l /cm/local/apps/slurm/current/lib64/slurm/spank_pyxis.so

systemctl mask rdma-ndd.service
mv /usr/lib/udev/rules.d/60-rdma-ndd.rules /usr/lib/udev/rules.d/60-rdma-ndd.rules.disabled

cat > /etc/rc.local <<- 'EOF'
#!/bin/bash
export PATH=/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/sbin:/usr/local/bin:$PATH

nics=($(lspci -D | grep -i eth|awk '{print $1}'|while read i; do basename $(ls -l /sys/class/net/|grep -o ${i}/.*); done))
for i in ${nics[*]}; do ip link set ${i} up; done
lldpcli configure system hostname .
lldpcli configure lldp portidsubtype ifname
lldpcli configure system interface pattern eth*,eno*,enp*,ens*,enP*
systemctl restart lldpd
# lldpcli show neighbors

start_rdma-ndd(){
  while : ; do
    if [[ $(lsmod | grep mlx5_ib | wc -l) -ne 0 && $(grep -r HCA- /sys/class/infiniband/mlx5_*/node_desc|wc -l) -ne 0 ]]; then
      break
    else
      sleep 10
    fi
  done
  pkill -9 rdma-ndd &>/dev/null
  sleep 10
  export RDMA_NDD_ND_FORMAT="%h %d"
  nohup /usr/sbin/rdma-ndd  -f --debug &
}
export -f start_rdma-ndd
nohup bash -c "start_rdma-ndd" &>/tmp/rdma-ndd.txt &

ipmitool raw 0x3c 0x74 100 &>/dev/null

#echo performance > /sys/module/pcie_aspm/parameters/policy

exit 0
EOF
chmod 755 /etc/rc.local

systemctl disable cachefilesd
rm -f /var/log/apt/*
rm -rf /tmp/*
truncate -s 0 /etc/machine-id
truncate -s 0 /root/.bash_history
exit 0

################################# Host Change - Need be done after slurm be installed
apt install -y bcm-post-install
image_name=gb200-image

#Update enroot.conf <--- Mimic post_install_tasks in $(dpkg -L bcm-post-install|grep slurm\.py$)
### dpkg -L $(dpkg -l|grep 'slurm..\..[0-9] '|awk '{print $2}')|grep enroot
#slurm23.02:
# /cm/shared/apps/slurm/var/etc/enroot.conf.jinja2 -> /cm/shared/apps/slurm/var/etc/enroot.conf
# /cm/shared/apps/slurm/var/cm/epilog-enroot.sh.jinja2 -> /cm/shared/apps/slurm/var/cm/epilog-enroot.sh
# /cm/shared/apps/slurm/var/cm/prolog-enroot.sh.jinja2 -> /cm/shared/apps/slurm/var/cm/prolog-enroot.sh
#slurm24.11: 
# /cm/local/apps/slurm/24.11/templates/enroot.conf.jinja2 -> /cm/shared/apps/slurm/etc/enroot.conf
# /cm/local/apps/slurm/24.11/templates/epilog-enroot.sh.jinja2 -> /cm/shared/apps/slurm/epilogs/epilog-enroot.sh
# /cm/local/apps/slurm/24.11/templates/prolog-enroot.sh.jinja2 -> /cm/shared/apps/slurm/prologs/prolog-enroot.sh

sed -i.ori \
  -e 's:^.*ENROOT_RUNTIME_PATH .*:ENROOT_RUNTIME_PATH  /raid/local/containers/enroot-runtime/user-$(id -u) :g' \
  -e 's:^.*ENROOT_CACHE_PATH.*:ENROOT_CACHE_PATH /raid/local/containers/enroot-cache/group-$(id -g)/$(id -u):g' \
  -e 's:^.*ENROOT_DATA_PATH.*:ENROOT_DATA_PATH /raid/local/containers/enroot-data/user-$(id -u):g' \
  -e 's:^.*ENROOT_CONFIG_PATH.*:ENROOT_CONFIG_PATH ${HOME}/.config/enroot:g' \
  -e 's:^.*ENROOT_ROOTFS_WRITABLE.*:ENROOT_ROOTFS_WRITABLE yes:g' \
  /cm/shared/apps/slurm/etc/enroot.conf

#Slurm PMIX profile #on login node or bcm headnode
/bin/cp -f /cm/local/apps/bcm-post-install/etc/slurm_pmix /etc/profile.d/pmix.sh

#Update epilog-enroot.sh script
/bin/cp -f /cm/local/apps/bcm-post-install/etc/enroot_epilog /cm/shared/apps/slurm/epilogs/epilog-enroot.sh
chmod 755 /cm/shared/apps/slurm/epilogs/epilog-enroot.sh

#Update prolog-enroot.sh script
/bin/cp -f /cm/local/apps/bcm-post-install/etc/enroot_prolog /cm/shared/apps/slurm/prologs/prolog-enroot.sh
chmod 755 /cm/shared/apps/slurm/prologs/prolog-enroot.sh

cm-chroot-sw-img -s /cm/shared /cm/images/${image_name} <<- '__END__'
rm -f /cm/local/apps/slurm/var/epilogs/50-epilog-enroot.sh
ln -sf /cm/shared/apps/slurm/epilogs/epilog-enroot.sh /cm/local/apps/slurm/var/epilogs/50-epilog-enroot.sh
rm -f /cm/local/apps/slurm/var/prologs/50-prolog-enroot.sh
ln -sf /cm/shared/apps/slurm/prologs/prolog-enroot.sh /cm/local/apps/slurm/var/prologs/50-prolog-enroot.sh
rm -f /etc/enroot/enroot.conf
ln -sf /cm/shared/apps/slurm/etc/enroot.conf /etc/enroot/enroot.conf
ls -l /cm/local/apps/slurm/var/epilogs/50-epilog-enroot.sh /cm/local/apps/slurm/var/prologs/50-prolog-enroot.sh /etc/enroot/enroot.conf
__END__

################################# Image Change
image_name=gb200-image

#Load Slurm module at login
echo "module load -s slurm" > /cm/images/${image_name}/etc/profile.d/zz-slurm-module.sh

#Add enroot directories in DGX image
mkdir -p  /cm/images/${image_name}/raid/local/containers/{enroot-runtime,enroot-cache,enroot-data}
chmod 777 /cm/images/${image_name}/raid/local/containers/{enroot-runtime,enroot-cache,enroot-data}

#Enroot PyTorch hook
cp /cm/local/apps/bcm-post-install/etc/slurm_pytorch_hook /cm/images/${image_name}/etc/enroot/hooks.d/50-slurm-pytorch.sh
chmod 755 /cm/images/${image_name}/etc/enroot/hooks.d/50-slurm-pytorch.sh

#Slurm PMIX profile
cp /cm/local/apps/bcm-post-install/etc/slurm_pmix /cm/images/${image_name}/etc/profile.d/pmix.sh
chmod 755 /cm/images/${image_name}/etc/profile.d/pmix.sh

#Enroot MLNX profile. 
cp /cm/local/apps/bcm-post-install/etc/enroot_mlnx_env /cm/images/${image_name}/etc/enroot/environ.d/20-mlx.env
chmod 755 /cm/images/${image_name}/etc/enroot/environ.d/20-mlx.env
sed -i \
  -e 's:^MELLANOX_VISIBLE_DEVICES=.*:MELLANOX_VISIBLE_DEVICES=all:g' \
  -e 's:\(^OMPI_MCA_btl_tcp_if_include=.*\):#\1:g' \
  /cm/images/${image_name}/etc/enroot/environ.d/20-mlx.env

#Slurmd config
cat > /cm/images/${image_name}/etc/sysconfig/slurmd << EOF
PMIX_MCA_ptl=^usock
PMIX_MCA_psec=none
PMIX_SYSTEM_TMPDIR=/var/empty
PMIX_MCA_gds=hash
EOF

cat > /cm/images/${image_name}/etc/sudoers.d/cmsupport << EOF
# Allow members of group cmsupport to execute any command
#%cmsupport ALL=(ALL:ALL) ALL
%cmsupport ALL=NOPASSWD: ALL
EOF

sed -i.ori -e "s:131072:$[1024*1024]:g" /cm/images/${image_name}/etc/security/limits.d/91-cm-limits.conf
sed -i.ori -e "s:131072:$[1024*1024*1024]:g" /cm/images/${image_name}/etc/sysctl.d/90-cm-sysctl.conf

## Add imex for GB200
cat > /cm/images/${image_name}/etc/enroot/mounts.d/30-imex.fstab <<- EOF
/dev/nvidia-caps-imex-channels
/dev/nvidia-caps
EOF
cat > /cm/images/${image_name}/etc/modprobe.d/nvidia-imex.conf <<- EOF
options nvidia NVreg_CreateImexChannel0=1
EOF

kvers=($(ls -1 /cm/images/${image_name}/lib/modules))
#Ref:
# https://docs.nvidia.com/grace-perf-tuning-guide/os-settings.html
# https://docs.nvidia.com/grace-perf-tuning-guide/optimizing-io.html
# https://nvidia.github.io/grace-cpu-benchmarking-guide/platform/index.html
# https://www.kernel.org/doc/Documentation/admin-guide/kernel-parameters.txt
#ast.modeset=0 be removed from kernel parameters
## pci=pcie_bus_perf,noaer,realloc=on
cmsh <<- __END__
softwareimage
use ${image_name}
set kernelversion ${kvers[0]}
set enablesol yes
set solport ttyAMA0
set kernelparameters "rd.driver.blacklist=nouveau nouveau.modeset=0 earlyprintk=serial,ttyAMA0,115200,keep console=tty0 console=ttyAMA0,115200n8 nvme_core.multipath=n pcie_ports=native namespace.unpriv_enable=1 user_namespace.enable=1 systemd.unified_cgroup_hierarchy=0 systemd.legacy_systemd_cgroup_controller intremap=no_x2apic_optout pciehp.pciehp_debug=y crashkernel=1G-:512M cgroup_enable=memory swapaccount=1 iommu.passthrough=1 init_on_alloc=0 transparent_hugepage=madvise numa_balancing=disable acpi_power_meter.force_cap_on=y nvidia_drm.modeset=0 pcie_aspm=off pci=pcie_bus_perf,realloc=on"
set enablesol yes
set solport ttyAMA0
commit
kernelmodules
add bonding; add ib_umad; add mlx5_core; add raid0; add raid1
commit
__END__

#FIXME: nvidia.NVreg_EnableGpuFirmware=0 doesn't take effect with nvidia-driver-570-open. It says GSP must be enable for B200 and later.
cmsh -c 'watch task list'

## Check the initrd and vmlinuz used for tftp
ls -l /cm/images/${image_name}/{initrd.img,vmlinuz}  /tftpboot/images/${image_name}/



#!/bin/bash

# https://ncp.gitlab-master-pages.nvidia.com/mission-control/docs/rack-bring-up-install/latest/config-for-provisioning/mixed-architecture-setup.html

wget -c https://support2.brightcomputing.com/pre-built-images/11.25.08/aarch64/bcmn-ubuntu2404-11.0-rc.tar.gz
wget -c https://support2.brightcomputing.com/pre-built-images/11.25.08/aarch64/bcm-cm-shared-ubuntu2404-11.0-rc.tar.gz
wget -c https://support2.brightcomputing.com/pre-built-images/11.25.08/aarch64/bcni-ubuntu2404-11.0-rc.tar.gz
mkdir -p /cm/images/default-image-ubuntu2404-aarch64 /cm/shared-ubuntu2404-aarch64/ /cm/node-installer-ubuntu2404-aarch64
tar zxvf bcmn-ubuntu2404-11.0-rc.tar.gz -C /cm/images/default-image-ubuntu2404-aarch64
tar zxvf bcm-cm-shared-ubuntu2404-11.0-rc.tar.gz -C /cm/shared-ubuntu2404-aarch64/
tar zxvf bcni-ubuntu2404-11.0-rc.tar.gz -C /cm/node-installer-ubuntu2404-aarch64
cm-image --verbose create all --arch aarch64 --distro ubuntu2404 --add-only
cm-image --verbose create all --arch aarch64 --distro ubuntu2404 --add-archos


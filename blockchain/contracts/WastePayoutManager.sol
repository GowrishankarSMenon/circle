// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract WastePayoutManager {
    constructor() {}

    function sendToUser(address toUser) external payable {
        require(msg.value > 0, "No ETH sent");
        payable(toUser).transfer(msg.value);
    }

    function sendToDeliveryBoy(address toBoy) external payable {
        require(msg.value > 0, "No ETH sent");
        payable(toBoy).transfer(msg.value);
    }


}

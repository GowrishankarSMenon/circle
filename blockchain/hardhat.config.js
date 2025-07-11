require("@nomicfoundation/hardhat-ethers");

module.exports = {
  solidity: "0.8.20",
  networks: {
    hardhat: {

    // add this if you want to connect to your locally running node
    
      chainId: 1337,
    }
  },
};

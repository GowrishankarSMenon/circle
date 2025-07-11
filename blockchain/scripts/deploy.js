const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  console.log(`Deploying WastePayoutManager with account: ${deployer.address}`);

  const Factory = await hre.ethers.getContractFactory("WastePayoutManager");
  const contract = await Factory.deploy();
  await contract.waitForDeployment(); // ethers v6

  console.log(`WastePayoutManager deployed at: ${await contract.getAddress()}`);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });

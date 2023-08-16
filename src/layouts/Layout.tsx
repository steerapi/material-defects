import { Outlet, useNavigation } from "react-router-dom";
import { FileManager } from "../components/FileManager";
import { PairList } from "../components/PairList";

export function Layout() {
  const navigation = useNavigation();
  return (
    <>
      <FileManager></FileManager>
      <PairList></PairList>
      <Outlet />
    </>
  );
}
